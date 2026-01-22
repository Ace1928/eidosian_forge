import torch
from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes.tools_common import legalize_graph
import itertools
import operator
from typing import Dict, List, Tuple
def split_result_tensors(result: torch.Tensor, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """
    A free function for use in the merge_matmul graph transformation below that
    splits the output from a merged matmul into the individual results for each
    input tensor.

    Arguments:
        result: The merged matmul result tensor.
        inputs: The list of inputs that were merged into one for the matmul.

    Returns:
        List of matmul results for each input tensor.
    """
    if isinstance(result, torch.fx.Proxy):
        splits = [0] * len(inputs)
    else:
        splits = [x.shape[0] for x in inputs]
    return torch.split(result, splits)