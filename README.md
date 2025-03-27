# SEA-GNN: SEQUENCE EXTENSION AUGMENTED GRAPH NEURAL NETWORK FOR SEQUENTIAL RECOMMENDATION
This is our TensorFlow implementation for the paper:
[SEA-GNN: Sequence Extension Augmented Graph Neural Network for Sequential Recommendation](https://ieeexplore.ieee.org/abstract/document/10446590/)


## Environment Setup
### Clone this directory
```
git clone https://github.com/ZuGeYunQian/SEA-GNN.git
```

### Install Dependencies
```
conda env create -f reco_env.yaml
conda activate recp_full_408
pip install -r reco_req.txt
```

### Dataset Preparation
We have processed the data in advance and put it in the resources folder.
Here we convert the original MoviceLens dataset into .csv format.
You can run it directly. The rest of the data preprocessing is in the code.


## Basic Usage
### Training and Testing
Run the files in examples/00_quick_start

## References
Please cite our paper if you use this repository.
```
@inproceedings{zu2024sea,
  title={SEA-GNN: Sequence Extension Augmented Graph Neural Network for Sequential Recommendation},
  author={Zu, Geyunqian and Zhao, Shengjie and Zeng, Jin and Dong, Shilong and Chen, Zixuan},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7090--7094},
  year={2024},
  organization={IEEE}
}
```

## Misc
The implementation is based on [Microsoft Recommender](https://github.com/microsoft/recommenders) and [SIGIR21-SURGE](https://github.com/tsinghua-fib-lab/SIGIR21-SURGE).
