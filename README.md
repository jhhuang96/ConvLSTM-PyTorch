

# ConvLSTM-Pytorch

## ConvRNN cell

Implement ConvLSTM/ConvGRU cell with Pytorch. This idea has been proposed in this paper: [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)

## Experiments with ConvLSTM on MovingMNIST

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

## Moving Mnist Generator

The script ``data/mm.py`` is the script to generate customized Moving Mnist based on [MNIST](http://yann.lecun.com/exdb/mnist/). 

```python
MovingMNIST(is_train=True,
            root='data/',
            n_frames_input=args.frames_input,
            n_frames_output=args.frames_output,
            num_objects=[3])
```

- is_train: If True, use script to generate data. If False, directly use Moving Mnist data  downloaded from http://www.cs.toronto.edu/~nitish/unsupervised_video/
- root: The path of MNIST data
- n_frames_input: Number of input frames (int)
- n_frames_output: Number of output frames (int)
- num_objects:  Number of digits in a frame (List) . [3] means there are 3 digits in each frame

## Result

 ![Result](https://github.com/jhhuang96/ConvLSTM-PyTorch/tree/master/images/movingmnist.png)

- The first line is the real data for the first 10 frames
- The second line is prediction of the model for the last 10 frames

## Citation

```
@inproceedings{xingjian2015convolutional,
  title={Convolutional LSTM network: A machine learning approach for precipitation nowcasting},
  author={Xingjian, SHI and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-Kin and Woo, Wang-chun},
  booktitle={Advances in neural information processing systems},
  pages={802--810},
  year={2015}
}
@inproceedings{xingjian2017deep,
    title={Deep learning for precipitation nowcasting: a benchmark and a new model},
    author={Shi, Xingjian and Gao, Zhihan and Lausen, Leonard and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun},
    booktitle={Advances in Neural Information Processing Systems},
    year={2017}
}
```