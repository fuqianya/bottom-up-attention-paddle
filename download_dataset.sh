# !/usr/bin/bash
# This script downloads the COCO2014 image captioning datasets and bottom-up-attention features.

# captions
echo "Downloading COCO captions ... "
wget -O ./data/caption_datasets.zip https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

echo "Unzipping the captions ... "
unzip ./data/caption_datasets.zip -d ./data

rm -f ./data/caption_datasets.zip

# features
echo "Downloading bottom-up features ... "
wget -O ./data/trainval_36.zip https://storage.googleapis.com/up-down-attention/trainval_36.zip

echo "Unzipping the features"
unzip ./data/trainval_36.zip -d ./data

rm -f ./data/trainval_36.zip
rm -rf ./data/trainval_36
