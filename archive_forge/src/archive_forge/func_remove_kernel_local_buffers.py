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
def remove_kernel_local_buffers(self):
    """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.
        """
    fused_node_names = V.kernel.store_buffer_names
    names_to_remove = []
    for out_buf in V.kernel.store_buffer_names:
        users = self.name_to_node[out_buf].users
        assert users is not None
        users = {user.get_name() for user in users if not user.is_weak}
        if users.issubset(fused_node_names):
            names_to_remove.append(out_buf)

    def remove_filter(n):
        return n not in V.kernel.must_keep_buffers and n not in V.kernel.args.input_buffers and (n not in self.mutation_renames) and (n not in self.mutation_real_name)
    names_to_remove = list(filter(remove_filter, names_to_remove))
    for name in names_to_remove:
        if name in V.kernel.args.inplace_buffers:
            buf = V.kernel.args.inplace_buffers[name]
            if isinstance(buf, str) and buf.startswith('REMOVED'):
                continue
            remove = all((n in names_to_remove for n in buf.other_names))
            if remove:
                self.remove_inplace_buffer(name)
            V.kernel.inplaced_to_remove.add(name)
        else:
            self.remove_buffer(name)