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
def create_foreach_nodes(self):
    removed_node_names = set()
    fe_nodes = []
    kept_node_names = self.name_to_fused_node.keys()
    for names in V.graph.lists.values():
        names = [name for name in names if name in kept_node_names and (not isinstance(self.name_to_node[name], NopKernelSchedulerNode))]
        if not names:
            continue
        removed_node_names.update(names)
        snodes = [self.name_to_node[name] for name in names]
        fe_node = ForeachKernelSchedulerNode(self, snodes)
        fe_nodes.append(fe_node)
        for name in names:
            self.name_to_fused_node[name] = fe_node
    self.nodes = [node for node in self.nodes if node.get_name() not in removed_node_names] + fe_nodes