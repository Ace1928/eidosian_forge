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
def warn_mix_layout(self, kernel_name):
    """
        Print message if the kernel have mixed layout inputs.
        Only care about 4D tensor for now.
        """
    if len(self.args.input_buffers) == 1 and len(self.args.output_buffers) == 1 and (len(self.args.inplace_buffers) == 0):
        return
    argdefs, call_args, signature = self.args.python_argdefs()
    uniform_stride_order = None
    for arg_name in call_args:
        buf = V.graph.get_buffer(arg_name)
        if buf and len(buf.layout.size) == 4:
            if len([x for x in buf.layout.size if x == 1]) == 3:
                continue
            stride_order = ir.get_stride_order(buf.layout.stride)
            if uniform_stride_order is None:
                uniform_stride_order = stride_order
            elif uniform_stride_order != stride_order:
                msg = yellow_text(f'Expected stride order {uniform_stride_order}, but found stride order' + f' {stride_order} for kernel {kernel_name}')
                log.warning(msg)
                stride_order_list = [ir.get_stride_order(V.graph.get_buffer(name).layout.stride) if V.graph.get_buffer(name) else None for name in call_args]
                size_list = [V.graph.get_buffer(name).layout.size if V.graph.get_buffer(name) else None for name in call_args]
                source_list = ['GraphInput' if name in V.graph.graph_inputs else 'IntermediateBuffer' if name in V.graph.name_to_buffer else None for name in call_args]
                msg = yellow_text(f'  param names {argdefs}\n  buf names {call_args}\n  strides {stride_order_list}' + f'\n  sizes {size_list}\n  sources {source_list}\n')
                log.warning(msg)
                return
    msg = green_text(f'All the inputs for the triton kernel {kernel_name} have uniform layout')
    log.warning(msg)