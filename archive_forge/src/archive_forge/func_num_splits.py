import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
@staticmethod
def num_splits(device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges, reduction_type, reduction_numel, input_node: Optional[IRNode]=None):

    def _is_static(x):
        return isinstance(x, (int, sympy.Integer))
    reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
    numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))
    should_split = is_triton(device) and reduction_type not in {'argmax', 'argmin'} and config.split_reductions and _is_static(reduction_numel_hint) and _is_static(numel_hint)
    if not should_split:
        return (ReductionHint.DEFAULT, 1)
    device_interface = get_interface_for_device(get_device_type(device))
    num_sm = device_interface.Worker.get_device_properties(device).multi_processor_count
    min_elements_per_thread = 32
    max_elements_per_thread = 512
    threads_per_sm = 2048
    min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
    max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm

    def inner_reduction_splits(reduction_numel_hint, numel_hint):
        num_warps = 8
        num_threads = 32 * num_warps
        if numel_hint >= 2 * num_sm:
            return 1
        if reduction_numel_hint <= 8192:
            return 1
        if reduction_numel_hint * numel_hint <= min_elements_per_device:
            split_size = min_elements_per_thread
        elif reduction_numel_hint * numel_hint < max_elements_per_device:
            target_blocks = num_sm * threads_per_sm // (2 * num_threads)
            blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
            tmp_split_size = (reduction_numel_hint + num_threads * blocks_per_output - 1) // (num_threads * blocks_per_output)
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
            if abs(closest - tmp_split_size) < 30:
                split_size = max(closest, min_elements_per_thread)
            else:
                split_size = tmp_split_size
        else:
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
            if abs(closest - max_elements_per_thread) < 50:
                split_size = closest
            else:
                split_size = max_elements_per_thread
        return (reduction_numel_hint + split_size * num_threads - 1) // (split_size * num_threads)

    def outer_reduction_splits(reduction_numel_hint, numel_hint):
        num_warps = 8
        num_threads = num_warps * 32
        rvals_per_thread = 4
        xvals_per_block = 128
        xblocks = (numel_hint + xvals_per_block - 1) // xvals_per_block
        if reduction_numel_hint * numel_hint < min_elements_per_device:
            split_size = min_elements_per_thread
        elif reduction_numel_hint * numel_hint < max_elements_per_device:
            target_blocks = num_sm * threads_per_sm // num_threads
            target_blocks = (target_blocks + xblocks - 1) // xblocks
            tmp_split_size = (reduction_numel_hint + rvals_per_thread * target_blocks - 1) // (rvals_per_thread * target_blocks)
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
            if abs(tmp_split_size - closest) < 20:
                split_size = max(closest, min_elements_per_thread)
            else:
                split_size = tmp_split_size
        else:
            divisors = sympy.divisors(reduction_numel_hint)
            closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
            if abs(closest - max_elements_per_thread) < 50:
                split_size = closest
            else:
                split_size = max_elements_per_thread
        return (reduction_numel_hint + rvals_per_thread * split_size - 1) // (rvals_per_thread * split_size)
    if numel_hint == 1:
        split = inner_reduction_splits(reduction_numel_hint, numel_hint)
        if split == 1:
            return (ReductionHint.INNER, split)
        if len(ranges) == 0 and input_node is not None and isinstance(input_node, TensorBox):
            new_ranges, new_reduction_ranges = extract_input_node_reduction_ranges(input_node)
            if new_ranges is not None and new_reduction_ranges is not None:
                extracted_numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(new_ranges + new_reduction_ranges))
                if reduction_numel_hint == extracted_numel_hint:
                    log.debug("Use previous IRNode's range and reduction_ranges instead of split. current ranges: %s, current reduction ranges: %s, current split: %d, new ranges: %s, new reduction ranges: %s", ranges, reduction_ranges, split, new_ranges, new_reduction_ranges)
                    return (ReductionHint.INNER, -1)
        return (ReductionHint.INNER, split)
    if reduction_numel_hint <= min_elements_per_thread or numel_hint >= num_sm * 2 * 32:
        return (ReductionHint.DEFAULT, 1)
    r = Reduction(device, dst_dtype, inner_fn, ranges, reduction_ranges, reduction_type, src_dtype, ReductionHint.DEFAULT)

    def get_read_indices(r):
        cb = ComputedBuffer(name=None, layout=FlexibleLayout(device=r.get_device(), dtype=r.get_dtype(), size=r.get_size()), data=r)
        read_writes = cb.get_read_writes()
        range_vars = [r for r in read_writes.range_vars if isinstance(r, sympy.Expr) and (not isinstance(r, sympy.Number))]
        indices = []
        changed = False
        for md in sorted(read_writes.reads, key=lambda x: x.name):
            if all((r in md.index.free_symbols for r in range_vars)):
                indices.append(md.index)
                if md.name in V.graph.name_to_buffer:
                    buf = V.graph.name_to_buffer[md.name]
                    original_stride = buf.layout.stride
                    buf.decide_layout()
                    if buf.layout.stride != original_stride:
                        changed = True
        return (indices, changed)
    indices, changed = get_read_indices(r)
    if changed:
        indices, _ = get_read_indices(r)
    if len(indices) == 0:
        return (ReductionHint.DEFAULT, 1)
    (_, reduction_vars), ranges = dependencies.index_vars_squeeze(r.get_size(), r.get_reduction_size())
    num_outer = 0
    num_inner = 0
    for i in indices:
        i = V.graph.sizevars.simplify_with_ranges(i, ranges)
        strides = V.graph.sizevars.stride_hints(i, reduction_vars, ranges.keys())
        outer = all((s > 1 for s in strides))
        if outer:
            num_outer += 1
        else:
            num_inner += 1
    if num_inner > num_outer:
        return (ReductionHint.INNER, inner_reduction_splits(reduction_numel_hint, numel_hint))
    else:
        return (ReductionHint.OUTER, outer_reduction_splits(reduction_numel_hint, numel_hint))