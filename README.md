
# LW-GNN
An offical PyTorch implementation of "Label-Wise Graph Convolutional Network for Heterophilic Graphs".
## Overview
* `./models`: This directory contains the proposed LW-GCN.
* `./dataset.py`: This is the code to load datasets.
* `./data`: The datasets are stored here.
* `./scripts`: It contains the scripts to reproduce the major reuslts of our paper.
* `./train_BI.py`: The program to train LW-GCN that incorporate the label-wise graph convolution and GCNII with model selection.
* `./train_DIS.py`: The program to train LW-GCN\G that removes the model selection module.
* `./log`: We provide our training log at ./log as a reference to reproduce the results in the paper

## Requirements

```
python>=3.7
torch==1.7.1
torch-geometric==1.7.2
cuda=11.0
```

### Install Packages

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch 
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.7.1
```

## Experiments
To reproduce the performance in the paper, you can run the bash files in the `.\scripts`. For example, to get results on cora datasets
```
# Run the LW-GCN on all datasets
bash train_BI.sh

# Run the LW-GCN\G that does not deploy model selection on all heterophilic graphs
bash train_DIS.sh
```

You can also check the our recorded log in `.\log` to help reproduce the experiments we conducted.

## Cite
If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{
dai2022labelwise,
title={Label-Wise Graph Convolutional Network for Heterophilic Graphs},
author={Enyan Dai and Shijie Zhou and Zhimeng Guo and Suhang Wang},
booktitle={Learning on Graphs Conference},
year={2022},
url={https://openreview.net/forum?id=HRmby7yVVuF}
}
```