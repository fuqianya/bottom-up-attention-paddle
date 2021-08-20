"""
train.py
~~~~~~~~

A script to eval the captioner.
"""
import os
import paddle
import json
import tqdm

from config.config import parse_opt
from model.captioner import Captioner
from model.dataloader import DataLoader
from pyutils.cap_eval import eval

opt = parse_opt()
assert opt.eval_model, 'please input eval_model'
assert opt.result_file, 'please input result_file'

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = paddle.load(opt.eval_model)
captioner = Captioner(chkpoint['idx2word'], chkpoint['settings'])
captioner.set_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}".
      format(opt.eval_model, chkpoint['epoch'], chkpoint['train_mode']))
captioner.eval()

captions = json.load(open(opt.captions, 'r'))
# process image captions before set up dataloader
print('\n====> process image captions begin')
word2idx = {}
for i, w in enumerate(chkpoint['idx2word']):
    word2idx[w] = i

captions_id = {}
for split, caps in captions.items():
    captions_id[split] = {}
    for img_id, seqs in tqdm.tqdm(caps.items(), ncols=100):
        tmp = []
        for seq in seqs:
            tmp.append([captioner.sos_id] +
                       [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                       [captioner.eos_id])
        captions_id[split][img_id] = tmp
captions = captions_id
print('\n====> process image captions end')

# set up dataloader
params = vars(opt)
params['pad_id'] = captioner.pad_id
dataloader = DataLoader(captions, params)

results = []
while True:
    data = dataloader.get_batch(split='test', seq_per_img=1)
    tmp = [data['img_ids'], data['fc_feats'], data['att_feats'], data['caps_tensor'],
           data['mask'], data['lengths'], data['groundtruth'], data['wrapped']]
    img_ids, fc_feats, att_feats, caps_tensor, mask_tensor, lengths, ground_truth, wrapped = tmp
    for i, img_id in enumerate(img_ids):
        fc_feat = fc_feats[i]
        att_feat = att_feats[i]
        with paddle.no_grad():
            rest, _ = captioner.sample(fc_feat, att_feat, beam_size=opt.beam_size, max_seq_len=opt.max_seq_len)
        results.append({'image_id': img_id, 'caption': rest[0]})
    if wrapped:
        dataloader.resetImageIerator(split='test')
        break

if not os.path.isdir(os.path.join(opt.result, opt.train_mode)):
    os.makedirs(os.path.join(opt.result, opt.train_mode))
json.dump(results, open(os.path.join(opt.result, opt.train_mode, opt.result_file), 'w'))

# evaluate the generated captions with bleu scores
eval.evaluate(opt.groundtruth, os.path.join(opt.result, opt.train_mode, opt.result_file))
