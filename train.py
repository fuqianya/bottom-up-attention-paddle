"""
train.py
~~~~~~~~

A script to train the captioner.
"""
import os
import json
import random
import numpy as np
from tqdm import tqdm

# paddle
import paddle
# options
from config.config import parse_opt
# model
from model.captioner import Captioner
# dataloader
from model.dataloader import DataLoader
# criterion
from model.loss import XECriterion, RewardCriterion
# utils
from utils.utils import get_ciderd_scorer, get_self_critical_reward


def main(params):
    train_mode = params['train_mode']  # xe or rl

    checkpoint_dir = os.path.join(params['checkpoint'], train_mode)
    result_dir = os.path.join(params['result'], train_mode)
    if not os.path.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.isdir(result_dir): os.makedirs(result_dir)

    idx2word = json.load(open(params['idx2word'], 'r'))
    captions = json.load(open(params['captions'], 'r'))

    # set random seed
    paddle.seed(params['seed'])
    random.seed(params['seed'])

    # set up model
    captioner = Captioner(idx2word, params['settings'])

    # process image captions before set up dataloader
    print('\n====> process image captions begin')
    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

    captions_id = {}
    for split, caps in captions.items():
        captions_id[split] = {}
        for img_id, seqs in tqdm(caps.items(), ncols=100):
            tmp = []
            for seq in seqs:
                tmp.append([captioner.sos_id] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [captioner.eos_id])
            captions_id[split][img_id] = tmp
    captions = captions_id
    print('\n====> process image captions end')

    # set up dataloader
    params['pad_id'] = captioner.pad_id
    dataloader = DataLoader(captions, params)

    # set up criterion
    xe_criterion = XECriterion()
    if train_mode == 'rl':
        rl_criterion = RewardCriterion()
        # prepare ciderd scorer
        ciderd_scorer = get_ciderd_scorer(captions, captioner.sos_id, captioner.eos_id)

    # set up optimizer
    lr = params['learning_rate']
    optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                      parameters=captioner.parameters(),
                                      grad_clip=paddle.fluid.clip.ClipGradByValue(params['grad_clip']))

    # load checkpoints for rl-training step
    if train_mode == 'rl':
        assert params['resume'], 'resume needs to provide when training with self-critical'
        print("====> loading checkpoint '{}'".format(params['resume']))
        checkpoint = paddle.load(params['resume'])
        assert params['settings'] == checkpoint['settings'], \
            'params[settings] and resume model settings are different'
        assert idx2word == checkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        captioner.set_state_dict(checkpoint['model'])
        print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}"
              .format(params['resume'], checkpoint['epoch'], checkpoint['train_mode']))

    def lossFun(dataloader, split='train', ss_prob=0.0):
        """Train or val for per epoch."""
        # set mode
        if split == 'train': captioner.train()
        else: captioner.eval()

        loss_val = 0.0
        reward_val = 0.0
        steps = 0
        while True:
            data = dataloader.get_batch(split)
            tmp = [data['img_ids'], data['fc_feats'], data['att_feats'], data['caps_tensor'],
                   data['mask'], data['lengths'], data['groundtruth'], data['wrapped']]
            img_ids, fc_feats, att_feats, caps_tensor, mask_tensor, lengths, ground_truth, wrapped = tmp

            if split == 'train' and train_mode == 'rl':
                sample_captions, sample_logprobs, seq_masks = \
                    captioner(fc_feats, att_feats, sample_max=0, max_seq_len=params['max_seq_len'], mode=train_mode)
                captioner.eval()
                with paddle.no_grad():
                    greedy_captions, _, _ = captioner(fc_feats, att_feats, sample_max=1,
                                                      max_seq_len=params['max_seq_len'], mode=train_mode)

                captioner.train()
                reward = get_self_critical_reward(sample_captions, greedy_captions, img_ids, ground_truth,
                                                  captioner.sos_id, captioner.eos_id, ciderd_scorer)
                loss = rl_criterion(sample_logprobs, seq_masks, paddle.to_tensor(reward))
                reward_val += float(np.mean(reward[:, 0]))
            else:
                # training with xe mode
                try:
                    pred = captioner(fc_feats, att_feats, caps_tensor, ss_prob=ss_prob)
                    loss = xe_criterion(pred, caps_tensor[:, 1:], mask_tensor)
                except:
                    print('corrupt data!')
                    continue

            loss_val += float(loss)
            steps += 1
            
            if split == 'train':
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
            
            if steps % params['losses_log_every'] == 0 and split == 'train':
                print('epoch: %d, step: %d, train_loss: %.4f, train_reward: %.4f'
                      % (epoch, steps, loss_val / steps, reward_val / steps))

            if wrapped:
                dataloader.resetImageIerator(split)
                if split == 'train': dataloader.shuffle_images(split)
                break

        return loss_val / steps, reward_val / steps

    previous_loss = None
    for epoch in range(params['max_epochs'] + 1):
        print('====> start epoch: %d' % epoch)

        # set up scheduled sampling probability
        ss_prob = 0.0
        if epoch > params['scheduled_sampling_start'] >= 0:
            frac = (epoch - params['scheduled_sampling_start']) // params['scheduled_sampling_increase_every']
            ss_prob = min(params['scheduled_sampling_increase_prob'] * frac, params['scheduled_sampling_max_prob'])

        # train the model for one epoch
        train_loss, train_reward = lossFun(dataloader, split='train', ss_prob=ss_prob)

        # eval the model
        with paddle.no_grad():
            val_loss, _ = lossFun(dataloader, split='val')

        # decay the learning rates
        if train_mode == 'xe' and previous_loss is not None and val_loss > previous_loss:
            lr = lr * 0.5
            print('epoch {} lr {}'.format(epoch, lr))
            optimizer.set_lr(lr)

        previous_loss = val_loss

        # save the model
        if epoch % params['save_checkpoint_every'] == 0:
            chkpoint = {
                'epoch': epoch,
                'model': captioner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': params['settings'],
                'idx2word': idx2word,
                'train_mode': train_mode,
            }
            chkpoint_path = os.path.join(checkpoint_dir, 'epoch_' + str(epoch) + '.pth')
            paddle.save(chkpoint, chkpoint_path)

        print('epoch: %d, train_loss: %.4f, train_reward: %.4f, val_loss: %.4f'
              % (epoch, train_loss, train_reward, val_loss))


if __name__ == '__main__':
    opt = parse_opt()
    params = vars(opt)  # convert to dict

    # call main()
    main(params)
