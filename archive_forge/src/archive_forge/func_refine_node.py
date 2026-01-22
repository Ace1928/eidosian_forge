from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
def refine_node(self, n: Node):
    """
        Returns a list of equality constraints for
        call_module and call_function nodes.
        Models the relation between input and output dimensions
        using constraints in case they are both tensors.
        All operations used in resnet50 are defined.
        """
    if n.type is None:
        n.type = Dyn
    n.type = self.replace_dyn_with_fresh_var(n.type)
    if n.op == 'call_function':
        if n.target in _REFINEMENT_RULES:
            self.constraints += _REFINEMENT_RULES[n.target](n)
        else:
            pass
    if n.op == 'call_module':
        module_instance = self.traced.get_submodule(n.target)
        if type(module_instance) in _REFINEMENT_RULES:
            self.constraints += _REFINEMENT_RULES[type(module_instance)](n)
        else:
            pass
    if n.op == 'output':

        def get_node_type(a):
            return a.type
        n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
        return n.type
    else:
        pass