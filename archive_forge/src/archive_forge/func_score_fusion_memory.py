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
def score_fusion_memory(self, node1, node2):
    """
        The first term in our fusion score that estimates number of saved memory operations.
        """
    common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (node2.read_writes.reads | node2.read_writes.writes)
    common_memory_deps = {dep for dep in common_memory_deps if not dep.has_unbacked_symbols()}
    return sum((dep.numbytes_hint() for dep in common_memory_deps))