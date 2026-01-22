import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
def eliminate_to_dtype(sub_graph: torch.fx.Graph):

    def _eliminate_duplicate_to_node(sub_graph: torch.fx.Graph):

        def _used_by_to(to_node: torch.fx.Node):
            return all((usr.target == 'to_dtype' for usr in to_node.users))
        all_to_nodes = [node for node in sub_graph.nodes if node.target == 'to_dtype']
        all_to_nodes_and_users = [{node: node.users} for node in all_to_nodes if _used_by_to(node)]
        for node_users in all_to_nodes_and_users:
            for node, users in node_users.items():
                if node in sub_graph.nodes and (all((usr.args[-1] == node.args[-1] for usr in users)) or (node in to_lowp_fp_legalized_nodes and all((usr.args[-1] in DTYPE_LOWP_FP for usr in users)))):
                    val_node = node.all_input_nodes[-1]
                    node.replace_all_uses_with(val_node)
                    sub_graph.erase_node(node)
        if sub_graph.owning_module is None:
            sub_graph.lint()
    _eliminate_duplicate_to_node(sub_graph)