# AutoGRN
The implementation code of the paper "AutoGRN: An Automated Graph Neural Network Framework for Gene Regulatory Network Inference".

## Requirements

```shell
python==3.7.12
torch==1.12.1+cu116
dgl==1.0.1.cu116
scikit-learn==1.0.2
numpy==1.19.5
networkx==2.6.3
requests==2.31.0
psutil==6.1.0
tqdm==4.67.1
pandas==1.3.5
tables==3.7.0
```


## Dataset Preparation

download the datasets from Google Drive:
```shell
https://drive.google.com/file/d/1nZai2lTdVmb-WwZIsY1e_mjxqoB4gkod/view?usp=sharing
```

then unzip this folder and post it in the root directory of the project (`AutoGRN/`)
```shell
unzip data_evaluation.zip
```

## Quick Start

A quick start example is given by:

```shell
$ python auto_main.py
```
### Optimal GNN Architectures

The following table summarizes the optimal GNN architectures searched by AutoGRN for each dataset:

| Dataset     | Conv Func   | Bi Conv Func  | Activation   | Hidden Dim |   Fusion Type    |
|-------------|-------------|---------------|--------------|------------|------------------|
| boneMarrow  | GATv2Conv   | BiGCNConv     | leaky_relu   | 512        | concat           |
| mESC_1      | SAGEConv    | BiNoneConv    | relu         | 256        | concat           |
| mESC_2      | TAGConv     | BiGraphConv   | leaky_relu   | 256        | sum              |
| mHSC_E      | SAGEConv    | BiSAGEConv    | leaky_relu   | 256        | abs_difference   |
| mHSC_GM     | GATv2Conv   | BiSAGEConv    | relu         | 1024       | max              |
| mHSC_L      | SAGEConv    | BiSAGEConv    | leaky_relu   | 256        | concat           |


Run ```auto_test.py``` to directly verify the result of the optimal GNN searched by AutoGRN
```shell
$ python auto_test.py
```
