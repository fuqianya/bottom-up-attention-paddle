"""
captioner.py
~~~~~~~~~~~~

Construct the image captioner with bottom-up and top-down attention that proposed in the paper.
"""
from collections import namedtuple

# paddle
import paddle
import paddle.nn as nn
import paddle.fluid as fluid

BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq'])

class Attention(nn.Layer):
    """Implementation of the bottom up and top down attention."""
    def __init__(self, settings):
        super(Attention, self).__init__()
        self.rnn_hid_dim = settings['rnn_hid_dim']  # number of hidden neurons of lstm
        self.att_hid_dim = settings['att_hid_dim']  # dim of the embedded attended feature

        self.h2att = nn.Linear(self.rnn_hid_dim, self.att_hid_dim)
        self.alpha_net = nn.Linear(self.att_hid_dim, 1)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, h, att_feats, pre_att_feats):
        """Compute up down attention and then obtain attended image feature.

        Inputs:
         - h: hidden state of the attention lstm, paddle.FloatTensor of shape [batch_size, rnn_hid_dim].
         - att_feats: embeded att feats, paddle.FloatTensor of shape [batch_size, num_objects, feat_emb_dim].
         - pre_att_feats: embeded att feats (used to obtain attended image feature). [batch_size, num_objects, att_hid_dim]
        """
        num_objects = att_feats.shape[1]

        # compute attention weights
        # see equation (3) in the paper for details
        h = self.h2att(h)  # [batch_size, att_hid_dim]
        h = fluid.layers.unsqueeze(h, axes=1)
        h = fluid.layers.expand_as(h, pre_att_feats)  # [batch_size, num_objects, att_hid_dim]
        pre_att_feats = paddle.tanh(h + pre_att_feats)  # [batch_size, num_objects, att_hid_dim]

        # [batch_size * num_objects, att_hid_dim]
        pre_att_feats = pre_att_feats.reshape((-1, self.att_hid_dim))
        weight = self.alpha_net(pre_att_feats)  # [batch_size * num_objects, 1]
        weight = weight.reshape((-1, num_objects))  # [batch_size, num_objects]

        # equation (4)
        weight = self.softmax(weight)  # [batch_size, num_objects]
        # equation (5)
        att_res = paddle.bmm(fluid.layers.unsqueeze(weight, axes=1), att_feats)
        att_res = fluid.layers.squeeze(att_res, axes=[1])  # [batch_size, feat_emb_dim]

        return att_res

class Captioner(nn.Layer):
    """Implementation of the image captioner proposed in the paper."""
    def __init__(self, idx2word, settings):
        super(Captioner, self).__init__()
        self.idx2word = idx2word
        self.vocab_size = len(idx2word)

        # index of special tokens
        self.pad_id = idx2word.index('<PAD>')
        self.unk_id = idx2word.index('<UNK>')
        self.sos_id = idx2word.index('<SOS>') if '<SOS>' in idx2word else self.pad_id
        self.eos_id = idx2word.index('<EOS>') if '<EOS>' in idx2word else self.pad_id

        # model settings
        self.dropout_prob = settings['dropout_p']  # probability of the dropout
        self.word_emb_dim = settings['word_emb_dim']  # dim of the word embedding
        self.fc_feat_dim = settings['fc_feat_dim']  # dim of input fc feature
        self.att_feat_dim = settings['att_feat_dim']  # dim of input att feature
        self.feat_emb_dim = settings['feat_emb_dim']  # dim of the embedded feature
        self.rnn_hid_dim = settings['rnn_hid_dim']  # number of hidden neurons of lstm
        self.att_hid_dim = settings['att_hid_dim']  # dim of the embedded attended feature

        # word embedding layer
        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size, self.word_emb_dim, padding_idx=self.pad_id),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout_prob))

        # fc feature embedding layer
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_dim, self.feat_emb_dim),
                                      nn.ReLU(),
                                      nn.Dropout(self.dropout_prob))

        # att feature embedding layer
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_dim, self.feat_emb_dim),
                                       nn.ReLU(),
                                       nn.Dropout(self.dropout_prob))

        # attention LSTM
        self.att_lstm = nn.LSTMCell(self.rnn_hid_dim + self.feat_emb_dim + self.word_emb_dim,
                                    self.rnn_hid_dim)

        # top down attention
        self.cxt2att = nn.Linear(self.feat_emb_dim, self.att_hid_dim)
        self.attention = Attention(settings)

        # languaeg LSTM
        self.lang_lstm = nn.LSTMCell(self.feat_emb_dim + self.rnn_hid_dim,
                                     self.rnn_hid_dim)

        # dropout for output
        self.lang_dropout = nn.Dropout(self.dropout_prob)

        # word classifier
        self.classifier = nn.Linear(self.rnn_hid_dim, self.vocab_size)

    def forward(self, *inputs, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']

        # forward_xe or forward_rl depend on the mode
        return getattr(self, 'forward_' + mode)(*inputs, **kwargs)

    def _init_hidden(self, batch_size):
        """Init hidden state and cell memory for lstm."""
        return (paddle.zeros([2, batch_size, self.att_lstm.hidden_size]),
                paddle.zeros([2, batch_size, self.att_lstm.hidden_size]))

    def _prepare_features(self, fc_feats, att_feats):
        """Embed both fc_feats and att_feats, and prepare att_feats for computing attention later.

        Inputs:
         - fc_feats: paddle.FloatTensor of shape [batch_size, fc_feat_dim].
         - att_feats: paddle.FloatTensor of shape [batch_size, num_objects, att_feat_dim].
        """
        # embed both fc_feats and att_feats into another space
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)

        # prepare feats for computing attention later
        pre_att_feats = self.cxt2att(att_feats)

        return fc_feats, att_feats, pre_att_feats

    def _forward_step(self, it, fc_feats, att_feats, pre_att_feats, state):
        """Forward the LSTM each time step.

        Inputs:
         - it: previous generated words. paddle.LongTensor of shape [batch_size, hidden_size].
         - fc_feats: paddle.FloatTensor of shape [batch_size, fc_feat_dim].
         - att_feats: paddle.FloatTensor of shape [batch_size, num_objects, att_feat_dim].
         - pre_att_feats: paddle.FloatTensor of shape [batch_size, num_objects, att_hid_dim]
         - state: hidden state and memory cell of lstm.
        """
        word_embs = self.word_embed(it)
        prev_h = state[0][1]

        # the input vector to the attention lstm consists of the previous output of language lstm (prev_h),
        # mean-pooled image feature (fc_feats) and an encoding of the previous generated word (word_embs).
        # see equation (2) in the paper for details.
        att_lstm_input = paddle.concat([prev_h, fc_feats, word_embs], 1)
        _, (h_att, c_att) = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        # then compute up down attention based on h_att and att_feats,
        # and obtain attended features by weighting average the pre_att_weights.
        # see equation (3), (4) and (5) in the paper for details
        att_res = self.attention(h_att, att_feats, pre_att_feats)  # [batch_size, att_feat_dim]

        # the input to the language lstm consists of the attended image feature (att_res) and
        # the output of the attention lstm.
        # see equation (6) for detials
        lang_lstm_input = paddle.concat([att_res, h_att], 1)
        _, (h_lang, c_lang) = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        # equation (7)
        output = self.lang_dropout(h_lang)  # [batch_size, rnn_hid_dim]
        output = self.classifier(output)  # [batch_size, vocab_size]
        logprobs = nn.functional.log_softmax(output, axis=1)  # [batch_size, vocab_size]
        
        state = (paddle.concat([fluid.layers.unsqueeze(h_att, 0), fluid.layers.unsqueeze(h_lang, 0)]),
                 paddle.concat([fluid.layers.unsqueeze(c_att, 0), fluid.layers.unsqueeze(c_lang, 0)]))
        
        return output, logprobs, state

    def forward_xe(self, fc_feats, att_feats, captions, ss_prob=0):
        """Train the captioner with cross-entropy loss.

        Inputs:
         - fc_feats: paddle.FloatTensor of shape [batch_size, fc_feat_dim].
         - att_feats: paddle.FloatTensor of shape [batch_size, num_objects, att_feat_dim].
         - captions: paddle.LongTensor of shape [batch_size, max_length].
         - ss_prob: float number that indicates the prob to sample words from generated words.
        """
        batch_size = fc_feats.shape[0]

        # prepare feats
        fc_feats, att_feats, pre_att_feats = self._prepare_features(fc_feats, att_feats)
        # init lstm state
        state = self._init_hidden(batch_size)

        logit_outputs = []
        prob_outputs = []
        # This is because we add start and end token into the caption
        for i in range(captions.shape[1] - 1):
            if self.training and i >= 1 and ss_prob > 0.0:
                # using scheduled sampling
                sample_prob = fluid.layers.uniform_random(shape=(batch_size, ), min=0, max=1)
                sample_mask = sample_prob < ss_prob
                if sample_mask.sum() == 0:
                    it = captions[:, i].clone()  # [batch_size, ]
                else:
                    sample_ind = sample_mask.nonzero().reshape((-1, ))
                    it = captions[:, i].clone()  # [batch_size, ]
                    prob_prev = prob_outputs[i - 1].detach().exp()

                    index_selected = paddle.index_select(x=paddle.multinomial(prob_prev, num_samples=1).reshape((-1, )),
                                                         index=sample_ind, axis=0)
                    assert index_selected.shape[0] == sample_ind.shape[0]
                    
                    # replace the groundtruth word with generated word when sampling next word
                    for j, ind in enumerate(sample_ind):
                        it[ind] = index_selected[j]
            else:
                it = captions[:, i].clone()  # [batch_size, ]
            output, logprobs, state = self._forward_step(it, fc_feats, att_feats, pre_att_feats, state)
                
            # used for compute loss
            logit_outputs.append(output)
            # used for sample words
            prob_outputs.append(logprobs)

        # we concat the output when finish all time steps
        logit_outputs = paddle.stack(logit_outputs, axis=1)  # [batch_size, max_len, vocab_size]

        return logit_outputs

    def forward_rl(self, fc_feats, att_feats, sample_max, max_seq_len):
        """Train the captioner with self-critical reward using reinforcement learning (i.e. policy gradient).

        Inputs:
         - fc_feats: paddle.FloatTensor of shape [batch_size, fc_feat_dim].
         - att_feats: paddle.FloatTensor of shape [batch_size, num_objects, att_feat_dim].
         - sample_max: using argmax or sampling when select generated words.
         - max_seq_len: max length of captions, i.e., clip the captions on this value.
        """
        batch_size = fc_feats.shape[0]
        fc_feats, att_feats, pre_att_feats = self._prepare_features(fc_feats, att_feats)
        state = self._init_hidden(batch_size)

        it = paddle.zeros([batch_size, ], dtype='int64').fill(self.sos_id)  # start token
        seq = []
        seq_logprobs = []
        seq_masks = []

        unfinished = (it == self.sos_id).cast('float32')

        for t in range(max_seq_len):
            _, logprobs, state = self._forward_step(it, fc_feats, att_feats, pre_att_feats, state)

            if sample_max:
                sample_logprobs = paddle.max(logprobs, 1)
                it = paddle.argmax(logprobs, 1)
                it = it.reshape((-1,)).cast('int64')
            else:
                prob_prev = paddle.exp(logprobs)
                it = paddle.multinomial(prob_prev, 1)
                it = it.reshape((-1,)).cast('int64')
                # prepare data for paddle.gather_nd
                batch_size = it.shape[0]
                gather_index = paddle.zeros((batch_size, 2), dtype='int64')  # [batch_size, 2]
                gather_index[:, 0] = paddle.arange(batch_size)
                gather_index[:, 1] = it
                # gather the logprobs at sampled positions
                sample_logprobs = paddle.gather_nd(logprobs, gather_index)

            it = it * unfinished.cast(it.dtype)
            seq.append(it)
            seq_logprobs.append(sample_logprobs.reshape((-1, )))
            seq_masks.append(unfinished.cast('float32'))

            unfinished = unfinished.cast('float32') * (it != self.eos_id).cast('float32')
            if unfinished.sum() == 0:
                break

        # we concat the output when finish all time steps
        seq = paddle.stack(seq, axis=1).cast('int64')
        seq_logprobs = paddle.stack(seq_logprobs, axis=1)
        seq_masks = paddle.stack(seq_masks, axis=1)

        return seq, seq_logprobs, seq_masks

    def sample(self, fc_feat, att_feat, max_seq_len=16, beam_size=3, decoding_constraint=1):
        """Sampling words for evaluation.

        Inputs:
         - fc_feat: paddle.FloatTensor of shape [batch_size, fc_feat_dim].
         - att_feat: paddle.FloatTensor of shape [batch_size, num_objects, att_feat_dim].
         - max_seq_len: max length of sampling.
         - beam_size: size of beam search.
         - decoding_constraint: wheather generate the last step word.
        """
        # set mode
        self.eval()

        fc_feat = fc_feat.reshape((1, -1))  # (1, fc_feat_dim)
        att_feat = att_feat.reshape((1, -1, att_feat.shape[-1]))  # [1, num_objects, att_feat_dim]

        fc_feat, att_feat, p_att_feat = self._prepare_features(fc_feat, att_feat)
        state = self._init_hidden(1)  # batch_size is 1

        # state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq
        candidates = [BeamCandidate(state, 0., [], self.sos_id, [])]
        for t in range(max_seq_len):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq = candidate
                if t > 0 and last_word_id == self.eos_id:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    it = paddle.to_tensor([last_word_id], dtype='int64')
                    _, logprobs, state = self._forward_step(it, fc_feat, att_feat, p_att_feat, state)  # 【1, vocab_size]
                    logprobs = logprobs.squeeze(0)  # [vocab_size, ]

                    # do not generate <PAD>, <SOS> and <UNK>
                    if self.pad_id != self.eos_id:
                        logprobs[self.pad_id] += float('-inf')
                        logprobs[self.sos_id] += float('-inf')
                        logprobs[self.unk_id] += float('-inf')

                    # do not generate last step word
                    # i.e. do not sample same words between adjacent time step
                    if decoding_constraint:
                        logprobs[last_word_id] += float('-inf')

                    output_sorted = paddle.sort(logprobs, descending=True)
                    index_sorted = paddle.argsort(logprobs, descending=True)
                    # beam search
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(state, log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id, word_id_seq + [word_id]))

            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag: break

        # captions, scores
        captions = [' '.join([self.idx2word[idx] for idx in candidate.word_id_seq if idx != self.eos_id])
                    for candidate in candidates]
        scores = [candidate.log_prob_sum for candidate in candidates]

        return captions, scores
