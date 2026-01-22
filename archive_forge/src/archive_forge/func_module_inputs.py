from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
def module_inputs(self) -> Sequence[torch.fx.Node]:
    """Extract module inputs from the sequence of fx nodes this instance holds.

        All node args that are produced by nodes outside of the module are considered module
        inputs. The order of returned module inputs is the same as the their use order.

        ### Known limitations

        The original ordering of module inputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module inputs.
        """
    nodes = list(self.fx_nodes())
    assert len(nodes) > 0, 'Cannot extract module inputs from empty nodes.'
    module_inputs: Dict[torch.fx.Node, None] = {}
    node_set: Set[torch.fx.Node] = set(nodes)

    def _extract_arg_if_node_outside_module(arg: Any):
        if isinstance(arg, torch.fx.Node) and arg not in node_set:
            module_inputs[arg] = None
    for node in nodes:
        pytree.tree_map(_extract_arg_if_node_outside_module, node.args)
        pytree.tree_map(_extract_arg_if_node_outside_module, node.kwargs)
    return list(module_inputs.keys())