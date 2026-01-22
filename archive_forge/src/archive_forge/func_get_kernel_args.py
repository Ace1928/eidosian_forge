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
def get_kernel_args(self, node_schedule, numel, reduction_numel):
    reductions = list(filter(lambda n: n not in (EnableReduction, DisableReduction) and n.is_reduction(), node_schedule))
    if len(reductions) > 0:
        hints = [self.reduction_hint(n) for n in reductions]
        if hints.count(hints[0]) == len(hints):
            reduction_hint_val = hints[0]
        else:
            reduction_hint_val = ReductionHint.DEFAULT
    else:
        reduction_hint_val = ReductionHint.DEFAULT
    mutations = set()
    for node in node_schedule:
        if hasattr(node, 'get_mutations'):
            mutations.update(node.get_mutations())
    index_dtype = self.select_index_dtype(node_schedule, numel, reduction_numel)
    return (reduction_hint_val, mutations, index_dtype)