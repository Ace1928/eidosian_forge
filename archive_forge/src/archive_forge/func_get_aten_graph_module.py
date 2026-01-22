import operator
import types
import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import (
from torch.nn.utils.fusion import fuse_conv_bn_weights
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
from torch.utils._pytree import LeafSpec
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import QuantizationAnnotation
def get_aten_graph_module(pattern: Callable, example_inputs: Tuple[Any, ...], is_cuda: bool=False, **kwargs) -> GraphModule:
    """
    Convert the pattern to an FX graph with decomposed aten ops.
    """
    if is_cuda:
        example_inputs = tuple([x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs])
    aten_pattern = capture_pre_autograd_graph(pattern, example_inputs, kwargs)
    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern