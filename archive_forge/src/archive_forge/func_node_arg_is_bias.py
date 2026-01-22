import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def node_arg_is_bias(node: Node, arg: Any) -> bool:
    """Returns if node arg is bias"""
    bias_index = None
    if 'target_dtype_info' in node.meta:
        bias_index = node.meta['target_dtype_info'].get('bias_index', None)
    if bias_index is not None and bias_index < len(node.args) and (node.args[bias_index] is arg):
        return True
    return node.kwargs.get('bias') is arg