#!/bin/bash

# Download SQuAD
echo "==> Downloading squad dataset..."
SQUAD_DIR=dataset/SQuAD
mkdir -p $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json

cd dataset; mkdir glove
cd glove

echo "==> Downloading glove vectors..."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip

echo "==> Unzipping glove vectors..."
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

echo "==> Done."

