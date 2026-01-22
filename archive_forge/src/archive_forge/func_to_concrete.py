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
def to_concrete(t):
    if isinstance(t, torch.Tensor):
        concrete = torch.ones_like(t, device='cpu')
        if concrete.dtype in [torch.float16, torch.float32, torch.float64, torch.int32]:
            concrete = concrete.to(torch.int64)
        return concrete
    return t