"""
dataloader_singe.py
~~~~~~~~~~~~~~~~~~

Implementing the dataloader to load batch of data for training and evaluation.
"""
import os
import random
import numpy as np

# paddle
import paddle

class DataLoader(object):
    def __init__(self, captions, params):
        self.params = params
        self.pad_id = params['pad_id']
        self.feat_dir = params['feat_dir']
        self.batch_size = params['batch_size']
        self.seq_per_img = params['seq_per_img']
        self.max_seq_len = params['max_seq_len'] + 1

        self.img_split_ix = {}
        self.img_iterators = {}
        # comstruct mapping from img_id to its captions
        self.captions = {}
        for split, caps in captions.items():
            self.img_split_ix[split] = []
            self.img_iterators[split] = 0
            for img_id, cap in caps.items():
                self.img_split_ix[split] += [img_id]
                self.captions[img_id] = cap

        for k, v in self.img_split_ix.items():
            print('assigned %d images to split %s.' % (len(v), k))

        print('found %d captions in the dataset.' % sum(len(self.captions[img_id]) for img_id in self.captions))

    # shuffle image ids
    def shuffle_images(self, split):
        random.shuffle(self.img_split_ix[split])

    # reset img_iterators[split] to start
    def resetImageIerator(self, split):
        self.img_iterators[split] = 0

    def fetch_feats(self, img_id):
        feats = np.load(os.path.join(self.feat_dir, str(img_id) + '.npz'))
        fc_feat = feats['fc_feat']
        att_feat = feats['att_feat']

        return fc_feat, att_feat

    def fetch_seqs(self, img_id, seq_per_img=None):
        # fetch the sequence labels
        captions = self.captions[img_id]

        if seq_per_img is None:
            return captions

        ncap = len(captions)

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seqs = []
            for q in range(seq_per_img):
                ix = random.randint(0, ncap-1)
                seqs.append(captions[ix])
        else:
            ix = random.randint(0, ncap - seq_per_img)
            seqs = captions[ix: ix + seq_per_img]

        return seqs

    def get_batch(self, split, seq_per_img=None):
        seq_per_img = seq_per_img or self.seq_per_img

        # split
        img_split_ix = self.img_split_ix[split]
        max_index = len(img_split_ix) - 1
        wrapped = False

        batch_img_ids = []
        batch_fc_feats = []
        batch_att_feats = []
        batch_seqs = []
        batch_length = []
        batch_mask = []
        groundtruth = {}
        for i in range(self.batch_size):
            ri = self.img_iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.img_iterators[split] = ri_next

            img_id = img_split_ix[ri]

            # fetch features
            fc_feat, att_feat = self.fetch_feats(img_id)
            for i in range(seq_per_img):
                batch_fc_feats.append(fc_feat)
                batch_att_feats.append(att_feat)
                batch_img_ids.append(img_id)

            # fetch all seqs for groundtruth
            groundtruth[img_id] = self.fetch_seqs(img_id)

            # fetch seq_per_img seqs for each image for training
            seqs = self.fetch_seqs(img_id, seq_per_img)

            for seq in seqs:
                length = min(self.max_seq_len, len(seq))
                batch_length.append(length)
                if len(seq) > self.max_seq_len:
                    batch_seqs.append(seq[:self.max_seq_len])
                    batch_mask.append([1] * (self.max_seq_len - 1))
                else:
                    padded_length = self.max_seq_len - len(seq)
                    batch_seqs.append(seq + [self.pad_id] * padded_length)
                    mask = [1] * (len(seq) - 1) + [0] * (self.max_seq_len - len(seq))
                    batch_mask.append(mask)

        # returned data
        data = {}
        data['img_ids'] = batch_img_ids
        data['fc_feats'] = np.array(batch_fc_feats, dtype='float32')
        data['att_feats'] = np.array(batch_att_feats, dtype='float32')
        data['caps_tensor'] = np.array(batch_seqs, dtype='int64')
        data['mask'] = np.array(batch_mask, dtype='float64')
        data['lengths'] = [l - 1 for l in batch_length]
        data['groundtruth'] = groundtruth
        data['wrapped'] = wrapped

        # turn all ndarray to paddle tensor
        data = {k: paddle.to_tensor(v, dtype=v.dtype) if type(v) is np.ndarray else v
                for k, v in data.items()}

        return data
