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
def dep_closure(node_name):
    reachable_names = {node_name}
    node = self.name_to_node[node_name]
    write_dep = next(iter(node.read_writes.writes))
    for read_dep in node.read_writes.reads:
        if read_dep.name in self.name_to_node and isinstance(read_dep, dependencies.MemoryDep) and isinstance(write_dep, dependencies.MemoryDep) and (read_dep.index == write_dep.index) and (read_dep.size == write_dep.size):
            reachable_names.update(dep_closure(read_dep.name))
    return reachable_names