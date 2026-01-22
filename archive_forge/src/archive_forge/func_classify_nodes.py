from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
def classify_nodes(joint_module):
    required_bw_nodes = set()
    for node in joint_module.graph.nodes:
        if node.op == 'placeholder' and 'tangents' in node.target:
            required_bw_nodes.add(node)
        if node in required_bw_nodes:
            for user in node.users:
                required_bw_nodes.add(user)
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    required_bw_nodes.update((o for o in bwd_outputs if o is not None))
    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, inputs, fwd_outputs)
    required_fw_nodes = {name_to_node[node.name] for node in forward_only_graph.nodes if node.op != 'output'}
    unclaimed_nodes = {node for node in joint_module.graph.nodes if node not in required_fw_nodes and node not in required_bw_nodes}
    return (fwd_outputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes)