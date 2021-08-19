"""
prepro.py
~~~~~~~~~

Preprocessing the COCO2014 dataset for training and evluation.
"""
import os
import sys
import csv
import json
import base64
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

def process_captions(params):
    """Prepare caption tokens, groundtruth sentences (test split) and vocab dict."""
    print('Start to process captions ... ')
    with open(params['coco_caption_file'], 'rb') as f:
        obj = json.load(f)
        images = obj['images']

    groundtruth = {}
    idx2word = Counter()
    captions = {'train': {}, 'val': {}, 'test': {}}
    for img in tqdm(images):
        split = 'train'
        if img['split'] == 'val':
            split = 'val'
        elif img['split'] == 'test':
            split = 'test'
        
        coco_id = img['cocoid']
        tokens, rows = [], []
        for sent in img['sentences']:
            tokens.append(sent['tokens'])
            # update the vocabulary
            idx2word.update(sent['tokens'])
            # only used for test split for evaluation
            rows.append(sent['raw'].lower().strip())

        captions[split][coco_id] = tokens
        if split == 'test':
            groundtruth[coco_id] = rows

    # dump captions into output_captions
    with open(params['output_captions'], 'w') as f:
        json.dump(captions, f)

    # dump groundtruth into output_groundtruth
    with open(params['output_groundtruth'], 'w') as f:
        json.dump(groundtruth, f)

    # filter the words that appear less than five times
    idx2word = idx2word.most_common()
    idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [w[0] for w in idx2word if w[1] > 4]
    print('Found %d words in the dataset' % len(idx2word))

    # dump idx2word into output_idx2word
    with open(params['output_idx2word'], 'w') as f:
        json.dump(idx2word, f)

    print('Process captions done!\n')

def prepare_features(params):
    """Prepare features for fast reading during training."""
    coco_feature_file = params['coco_feature_file']
    output_feature_dir = params['output_feature_dir']
    assert os.path.isfile(coco_feature_file), 'Feature file not Found!' \
                                              'Pleause run download_dataset.sh first!'

    csv.field_size_limit(sys.maxsize)
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

    print('Start to process features ... ')
    with open(coco_feature_file, "r") as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames = FIELDNAMES)
        for item in tqdm(reader):
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            
            item['features'] = np.frombuffer(base64.decodebytes(item['features'].encode()), dtype=np.float32).\
                reshape((item['num_boxes'], -1))

            save_path = os.path.join(output_feature_dir, str(item['image_id']))
            np.savez(save_path, att_feat=item['features'], fc_feat=item['features'].mean(0))

    print('Process features done!')

def main(params):
    output_feat_dir = params['output_feature_dir']
    if not os.path.isdir(output_feat_dir):
        os.makedirs(output_feat_dir)
        
    # prepare captions
    process_captions(params)

    # prepare features
    prepare_features(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input files
    parser.add_argument('--coco_caption_file', type=str, default='./data/dataset_coco.json',
                        help='path to the caption file of coco2014.')
    parser.add_argument('--coco_feature_file', type=str, default='./data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv',
                        help='path to the feature file of coco2014')
    # output files
    parser.add_argument('--output_feature_dir', type=str, default='./data/feats',
                        help='folder to be store the preprocessed features.')
    parser.add_argument('--output_captions', type=str, default='./data/captions.json',
                        help='path to store the preprocessed captions.')
    parser.add_argument('--output_idx2word', type=str, default='./data/idx2word.json',
                        help='path to store the word-index mapping dictionary.')
    parser.add_argument('--output_groundtruth', type=str, default='./data/groundtruth.json',
                        help='path to store the captions of test split, which prepares for evaluation.')

    opt = parser.parse_args()
    params = vars(opt)  # convert to dictionary

    # call main()
    main(params)
