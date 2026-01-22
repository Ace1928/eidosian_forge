import torch
from torch.fx import map_arg, Node
from torch.fx.graph import Graph
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.fx import GraphModule
from .utils import (
from ..utils import _parent_name
from ..qconfig import QConfigAny
from ..quantization_mappings import get_quantized_operator
from .utils import create_node_from_old_node_preserve_meta
from typing import Dict, Tuple, Type, List, Callable, Any, Union, Set, Optional
import operator
def is_general_tensor_shape_node(node, modules):
    func_list = [torch.narrow, torch.transpose, torch.repeat_interleave, torch.squeeze, torch.stack, torch.unsqueeze, torch.nn.functional.pixel_shuffle, torch.nn.functional.pixel_unshuffle]
    method_list = ['contiguous', 'detach', 'detach_', 'permute', 'repeat', 'repeat_interleave', 'reshape', 'resize_', 'shape', 'size', 'squeeze', 'squeeze_', 'transpose', 'unsqueeze', 'unsqueeze_', 'view']
    module_type_list = [torch.nn.Identity, torch.nn.PixelShuffle, torch.nn.PixelUnshuffle]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)