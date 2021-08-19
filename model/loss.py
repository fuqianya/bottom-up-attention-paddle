"""
loss.py
~~~~~~~

Loss functions used to compute loss.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class XECriterion(nn.Layer):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, mask):
        """Inputs:
         - pred: [batch_size, seq_len, vocab_size].
         - target: [batch_size, seq_len].
         - mask: [batch_size, seq_len].
        """
        loss_ = F.cross_entropy(pred, target, reduction='none')
        loss_ *= mask

        return paddle.sum(loss_) / paddle.sum(mask)

class RewardCriterion(nn.Layer):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq_logprobs, seq_masks, reward):
        output = - seq_logprobs * seq_masks * reward
        output = paddle.sum(output) / paddle.sum(seq_masks)

        return output
