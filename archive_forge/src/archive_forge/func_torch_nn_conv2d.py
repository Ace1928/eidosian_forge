import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
def torch_nn_conv2d(self, input):
    h_in, w_in = input.shape[-2:]
    shape = None
    padding = self.padding
    if padding == 'valid':
        padding = (0, 0)
    if padding == 'same':
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        h_out = math.floor((h_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor((w_in + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        shape[-2:] = [h_out, w_out]
    shape[-3] = self.out_channels
    return torch.empty(shape, device='meta')