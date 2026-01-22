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
def is_unaligned_buffer(self, buf_name):
    if buf_name in V.graph.graph_inputs or buf_name in V.graph.constants:
        return False
    node = self.name_to_node[buf_name]
    layout = node.node.get_layout()
    if isinstance(layout, ir.AliasedLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False