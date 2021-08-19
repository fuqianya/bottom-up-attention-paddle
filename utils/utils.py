"""
utils.py
~~~~~~~~

A module contains all utils used in this project.
"""
# CIDERD
from pyutils.self_critical.cider.pyciderevalcap.ciderD.ciderD import CiderD
# BLEU
from pyutils.self_critical.bleu.bleu import Bleu

import tqdm

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def _array_to_str(arr, sos_token, eos_token):
    """Convert a array to a string."""
    arr = list(arr)
    if arr[0] == sos_token:
        arr = arr[1:]

    out = ''
    for i in range(len(arr)):
        if arr[i] == eos_token:
            break
        out += str(arr[i]) + ' '
    out += str(eos_token)

    return out.strip()

def get_ciderd_scorer(split_captions, sos_token, eos_token):
    """Compute ciderd score for captions."""
    captions = {}
    for caps in split_captions.values():
        captions.update(caps)

    refs_idxs = []
    for caps in tqdm.tqdm(captions.values(), ncols=100):
        ref_idxs = []
        for cap in caps:
            ref_idxs.append(_array_to_str(cap, sos_token, eos_token))
        refs_idxs.append(ref_idxs)

    scorer = CiderD(refs=refs_idxs)
    return scorer

def get_self_critical_reward(sample_captions, greedy_captions, img_ids, ground_truth,
                             sos_token, eos_token, scorer):
    """Compute self-critical reward for optimization."""
    batch_size = len(img_ids)
    sample_captions = sample_captions.cpu().numpy()
    greedy_captions = greedy_captions.cpu().numpy()
    assert sample_captions.shape[0] == greedy_captions.shape[0] == batch_size
    max_seq_len = sample_captions.shape[1] + 1
    sample_result = []
    greedy_result = []
    gts = {}
    for i, img_id in enumerate(img_ids):
        sample_result.append({'image_id': img_id, 'caption': [_array_to_str(sample_captions[i], sos_token, eos_token)]})
        greedy_result.append({'image_id': img_id, 'caption': [_array_to_str(greedy_captions[i], sos_token, eos_token)]})
        caps = []
        for cap in ground_truth[img_id]:
            caps.append(_array_to_str(cap[:max_seq_len], sos_token, eos_token))
        gts[img_id] = caps
    all_result = sample_result + greedy_result
    if isinstance(scorer, CiderD):
        _, scores = scorer.compute_score(gts, all_result)
    elif isinstance(scorer, Bleu):
        _, scores = scorer.compute_score(gts, all_result)
        scores = np.array(scores[3])
    else:
        raise Exception('do not support this scorer: %s' % type(scorer))

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sample_captions.shape[1], 1)
    return rewards
