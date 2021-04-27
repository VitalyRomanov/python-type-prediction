# Python Type Prediction

This repository contains code for type prediction for python using FastText and GNN embedding.

## Dataset

The data can be found in `dataset` directory. Each entry is a json record with fields for source code text, type annotations, return type, and graaph references.

## Pretrained embeddings

Pretrained embeddings can be found in `pretrained_embeddings`. 

## Train model 

To train the model run

```bash
python type-prediction.py --data_path dataset/functions_with_annotations.jsonl --graph_emb_path pretrained_embeddings/graph_embeddings_dim_100.pkl --word_emb_path pretrained_embeddings/fasttext_codesearchnet_dim_100.pkl
```