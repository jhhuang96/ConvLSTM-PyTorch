import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import make_layers
import logging


def MNISTdataLoader(path):
    # load moving mnist data
    # data shape = [time steps, batch size, width, height] = [20, batch_size, 64, 64]
    data = np.load(path)
    train = data.transpose(1, 0, 2, 3)
    return train


class MovingMNISTdataset(Dataset):
    # dataset class for moving MNIST data
    def __init__(self, path):
        self.path = path
        self.data = MNISTdataLoader(path)

    def __len__(self):
        return len(self.data[:, 0, 0, 0])

    def __getitem__(self, indx):
        ## getitem method
        self.trainsample_ = self.data[indx, ...]
        self.sample_ = self.trainsample_ / 255.0

        self.sample = torch.from_numpy(np.expand_dims(self.sample_,
                                                      axis=1)).float()
        return self.sample


# input: B, C, H, W
# flow: [B, 2, H, W]
def warp(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()  # H*W
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()  # H*W
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)  # B,1,H,W
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)  # B,1,H,W
    grid = torch.cat((xx, yy), 1).float()  # B,2,H,W
    vgrid = grid + flow  # B,2,H,W

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # B,2,64,64 -> B,64,64,2
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output  # B,num_fliter,H,W


class BaseConvRNN(nn.Module):
    def __init__(self,
                 num_filter,
                 b_h_w,
                 h2h_kernel=(3, 3),
                 h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3),
                 i2h_stride=(1, 1),
                 i2h_pad=(1, 1),
                 i2h_dilate=(1, 1),
                 act_type=torch.tanh,
                 prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] -
                                  1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] -
                                  1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h)\
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                             // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0


class TrajGRU(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self,
                 input_channel,
                 num_filter,
                 b_h_w,
                 act_type,
                 zoneout=0.0,
                 L=5,
                 i2h_kernel=(3, 3),
                 i2h_stride=(1, 1),
                 i2h_pad=(1, 1),
                 h2h_kernel=(5, 5),
                 h2h_dilate=(1, 1)):
        super(TrajGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU')
        self._L = L
        self._zoneout = zoneout

        # correspond to wxz, wxr, wxh
        # reset_gate, update_gate, H’_t
        self.i2h = nn.Conv2d(in_channels=input_channel,
                             out_channels=self._num_filter * 3,
                             kernel_size=self._i2h_kernel,
                             stride=self._i2h_stride,
                             padding=self._i2h_pad,
                             dilation=self._i2h_dilate)

        # inputs to flow
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # hidden to flow
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # generate flow
        self.flows_conv = nn.Conv2d(
            in_channels=32,
            out_channels=self._L * 2,  # U_t,V_t
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2))

        # correspond to hh, hz, hr， 1 * 1 filter
        self.ret = nn.Conv2d(in_channels=self._num_filter * self._L,
                             out_channels=self._num_filter * 3,
                             kernel_size=(1, 1),
                             stride=1)

    # inputs: B*C*H*W
    def _flow_generator(self, inputs, states):
        # inputs:B,1,64,64 ; states:8,num_filter,64,64
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)  # B,32,64,64
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)  # B,32,64,64
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)

        flows = self.flows_conv(f_conv1)  # B,L*2,64,64
        flows = torch.split(flows, 2, dim=1)  # tuple: L*{B,2,64,64}
        return flows

    # inputs 和 states 不同时为空
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=10):
        if states is None:
            states = torch.zeros(
                (inputs.size(1), self._num_filter, self._state_height,
                 self._state_width),
                dtype=torch.float).cuda()  # b,num_filter,64,64
        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h = self.i2h(torch.reshape(
                inputs, (-1, C, H, W)))  # B*S,num_filter*3,64,64
            i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2),
                                      i2h.size(3)))  # S,B,num_filter*3,64,64
            i2h_slice = torch.split(i2h, self._num_filter,
                                    dim=2)  # tuple : 3* {S,B,num_filter,64,64}

        else:
            i2h_slice = None

        prev_h = states
        outputs = []
        for i in range(seq_len):
            if inputs is not None:
                flows = self._flow_generator(inputs[i, ...], prev_h)
            else:
                flows = self._flow_generator(None, prev_h)
            warpped_data = []
            for j in range(len(flows)):  # num of L
                flow = flows[j]  # B,2,64,64
                warpped_data.append(warp(prev_h, -flow))
            # List : 13 * {B,num_filter,64,64} -> B,13*num_filter,64,64
            warpped_data = torch.cat(warpped_data, dim=1)
            h2h = self.ret(warpped_data)
            h2h_slice = torch.split(h2h, self._num_filter, dim=1)
            if i2h_slice is not None:
                update_gate = torch.sigmoid(i2h_slice[0][i, ...] +
                                            h2h_slice[0])
                reset_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][i, ...] +
                                         reset_gate * h2h_slice[2])
            else:
                update_gate = torch.sigmoid(h2h_slice[0])
                reset_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            if self._zoneout > 0.0:
                mask = F.dropout2d(torch.zeros_like(prev_h), p=self._zoneout)
                next_h = torch.where(mask, next_h, prev_h)
            outputs.append(next_h)
            prev_h = next_h

        return torch.stack(outputs), next_h


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(rnns)
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1),
                                      input.size(2), input.size(3)))
        outputs_stage, state_stage = rnn(input, None)
        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        input = input.transpose(0, 1)  # S,B,C,H,W
        hidden_states = []
        logging.debug(input.size())
        for i in range(1, self.blocks + 1):
            input, state_stage = self.forward_by_stage(
                input, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=10)  # out_length = 10
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1),
                                      input.size(2), input.size(3)))
        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1],
                                      getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i - 1],
                                          getattr(self, 'stage' + str(i)),
                                          getattr(self, 'rnn' + str(i)))
        return input


class EF(nn.Module):
    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        output = output.transpose(0, 1)
        return output
