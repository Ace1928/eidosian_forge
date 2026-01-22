import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_group_weight_grad_sample(input, grad_output, weight_shape, stride, padding, dilation, batch_size, func):
    I = input.shape[1]
    O = grad_output.shape[1]
    input_ = input.transpose(0, 1)
    grad_output_ = grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, *grad_output.shape[2:])
    weight_grad_sample = func(input_, grad_output_, None, stride=dilation, padding=padding, dilation=stride, groups=batch_size)
    input_dims = conv_picker(func, 3, 4, 5)
    for i in range(2, input_dims):
        weight_grad_sample = weight_grad_sample.narrow(i, 0, weight_shape[i])
    weight_grad_sample = weight_grad_sample.view(I, batch_size, O, *weight_grad_sample.shape[2:])
    weight_grad_sample = weight_grad_sample.movedim(0, 2)
    return weight_grad_sample