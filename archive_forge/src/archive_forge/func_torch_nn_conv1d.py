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
def torch_nn_conv1d(self, input):
    l_in = input.shape[-1]
    shape = None
    padding = self.padding
    if padding == 'valid':
        padding = (0, 0)
    if padding == 'same':
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        l_out = math.floor((l_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        shape[-1] = l_out
    shape[-2] = self.out_channels
    return torch.empty(shape, device='meta')