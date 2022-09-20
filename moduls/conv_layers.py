import torch
import torch.nn as nn
from .common import *
def conv_layers(num_input_channels=416, num_output_channels=416, num_hidden=1000,need_sigmoid = True, need_tanh = False):
    model = nn.Sequential(
        nn.Conv2d(
            num_input_channels,
            416,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(416),
        nn.Conv2d(
            416,
            416,
            kernel_size=1,
            padding=0),
        nn.ReLU(),

        nn.Conv2d(
            416,
            416,
            kernel_size=1,
            padding=0),
        nn.ReLU(),

        nn.Conv2d(
            416,
            num_output_channels,
            kernel_size=1,
            padding=0),
    )
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())
    return model
