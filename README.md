

# ConvLSTM-Pytorch

## ConvRNN cell

Implement ConvLSTM/ConvGRU cell with Pytorch. This idea has been proposed in this paper: [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)

## Experiments with ConvLSTM on movingMNIST

Encoder-decoder structure. Takes in a sequence of 10 movingMNIST fames and attempts to output the remaining frames.

## Instructions

Requires `Pytorch v1.1` or later (and GPUs)

Clone repository

```
git clone https://github.com/jhhuang96/ConvLSTM-PyTorch.git
```

To run endoder-decoder network for prediction moving-mnist:

```python
python main.py
```

## Citation

```
@inproceedings{xingjian2017deep,
    title={Deep learning for precipitation nowcasting: a benchmark and a new model},
    author={Shi, Xingjian and Gao, Zhihan and Lausen, Leonard and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun},
    booktitle={Advances in Neural Information Processing Systems},
    year={2017}
}
```