import torch
import operator
import warnings
from typing import Callable, Dict, Iterable
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
from torch.fx.experimental.migrate_gradual_types.operation import \
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
def generate_constraints(self, counter=0):
    """
        Iterate through every node and generate constraints
        Effect: self.constraints will be populated with the final constraints
        """
    graph = self.graph
    all_constraints = []
    for n in graph.nodes:
        constraints, counter = self.generate_constraints_node(n, counter)
        all_constraints += constraints
    return (Conj(all_constraints), counter)