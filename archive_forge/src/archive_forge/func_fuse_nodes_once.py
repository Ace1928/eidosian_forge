import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
def fuse_nodes_once(self):
    """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuses(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
    fused_nodes = set(self.nodes)
    for node1, node2 in self.get_possible_fusions():
        node1 = self.name_to_fused_node[node1.get_first_name()]
        node2 = self.name_to_fused_node[node2.get_first_name()]
        if self.can_fuse(node1, node2) and (not self.will_fusion_create_cycle(node1, node2)):
            if not self.speedup_by_fusion(node1, node2):
                continue
            node3 = fuse(node1, node2)
            fused_nodes.remove(node1)
            fused_nodes.remove(node2)
            fused_nodes.add(node3)
            self.name_to_fused_node.update({n.get_name(): node3 for n in node3.get_nodes()})
    self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
    self.topological_sort_schedule()
    self.prune_redundant_deps()