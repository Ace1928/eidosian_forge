import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def generate_outputs(self):
    graph = self.add_node.graph
    for i, arg in enumerate(cast(Tuple[Any, ...], self.add_node.args[0])):
        with graph.inserting_after(self.add_node):
            updated_arg = graph.call_function(operator.getitem, (self.add_node, i))
        with graph.inserting_after(updated_arg):
            output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
        self.outputs.append(output_copy)
    assert self.outputs, f'The output for {self.add_node} is empty.'