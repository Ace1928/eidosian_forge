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
def torch_arange(*args, **kwargs):
    n = len(args)
    step = 1
    if n == 1:
        start = 0
        end = args[0]
    elif n == 2:
        start, end = args
    else:
        start, end, step = args
    if isinstance(start, float):
        start = int(start)
    if isinstance(end, float):
        start = int(end)
    if isinstance(step, float):
        step = int(step)
    step = kwargs.get('step', step)
    dtype = kwargs.get('dtype')
    return torch.empty((end - start) // step, dtype=dtype, device='meta')