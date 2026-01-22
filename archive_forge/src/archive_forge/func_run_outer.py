import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
def run_outer(self):
    i = 0
    for node in self.flat_graph.nodes:
        self.print(i, node.meta.get('nn_module_stack'), node.format_node())
        i += 1
    node_idx: int = 0
    node = self.nodes[node_idx]
    while node.op == 'placeholder':
        self.copy_node(node)
        node_idx += 1
        node = self.nodes[node_idx]
    self.run_from(node_idx)
    for node in self.flat_graph.nodes:
        if node.op == 'output':
            self.copy_node(node)