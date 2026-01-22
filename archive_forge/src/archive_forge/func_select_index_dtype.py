from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
@staticmethod
def select_index_dtype(node_schedule, numel, reduction_numel):
    buffer_names = set()
    for node in node_schedule:
        if not isinstance(node, scheduler.BaseSchedulerNode):
            continue
        buffer_names.update(node.get_names())
        buffer_names.update(node.used_buffer_names())

    def _get_buffer(name: str) -> Union[ir.Buffer, ir.TensorBox]:
        if name in V.graph.name_to_buffer:
            return V.graph.name_to_buffer[name]
        elif name in V.graph.graph_inputs:
            return V.graph.graph_inputs[name]
        elif name in V.graph.constants:
            data = V.graph.constants[name]
            return ir.ConstantBuffer(name, ir.FixedLayout(data.device, data.dtype, *V.graph.static_sizes_strides(data)))
        raise RuntimeError(f'Failed to find buffer matching name {name}')
    buffers = [_get_buffer(name) for name in buffer_names]
    total_numel = numel * reduction_numel
    if TritonScheduling.can_use_32bit_indexing(total_numel, buffers):
        return 'tl.int32'
    return 'tl.int64'