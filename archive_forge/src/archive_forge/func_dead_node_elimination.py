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
def dead_node_elimination(self):
    """
        Remove any nodes without users
        """
    again = True
    while again:
        updated_nodes = []
        for node in self.nodes:

            def can_eliminate_user(user: NodeUser):
                return user.is_weak or user.get_name() in V.graph.removed_buffers
            can_eliminate = not node.has_side_effects() and all((can_eliminate_user(u) for u in node.users))
            if not can_eliminate:
                updated_nodes.append(node)
            else:
                log.debug('removed dead node: %s', node.get_name())
                V.graph.removed_buffers.add(node.get_name())
        again = len(self.nodes) > len(updated_nodes)
        self.nodes = updated_nodes
    for node in self.nodes:
        node.prune_weak_deps()