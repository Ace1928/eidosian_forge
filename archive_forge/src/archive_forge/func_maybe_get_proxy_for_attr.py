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
def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
    for n, p in collection_to_search:
        if attr_val is p:
            if n not in parameter_proxy_cache:
                kwargs = {}
                if 'proxy_factory_fn' in inspect.signature(self.create_proxy).parameters:
                    kwargs['proxy_factory_fn'] = None if not self.param_shapes_constant else lambda node: ParameterProxy(self, node, n, attr_val)
                val_proxy = self.create_proxy('get_attr', n, (), {}, **kwargs)
                parameter_proxy_cache[n] = val_proxy
            return parameter_proxy_cache[n]
    return None