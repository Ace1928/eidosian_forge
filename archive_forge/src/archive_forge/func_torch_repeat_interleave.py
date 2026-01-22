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
def torch_repeat_interleave(*args, dim=None, output_size=None):
    num_args = len(args)
    if num_args == 1:
        shape = [output_size if output_size is not None else args[0].sum()]
    else:
        shape = list(args[0].shape)
        if dim is None:
            if num_args > 2:
                dim = args[2]
            else:
                shape = [sum(shape)]
                dim = 0
        repeats = args[1]
        if isinstance(repeats, int) or torch.numel(repeats) == 1:
            shape[dim] *= int(repeats)
        else:
            shape[dim] = output_size if output_size is not None else repeats.sum()
    return torch.empty(*shape, device='meta')