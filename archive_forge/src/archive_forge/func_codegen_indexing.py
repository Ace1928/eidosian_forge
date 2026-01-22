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
def codegen_indexing(self, expr: sympy.Expr):
    expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
    for sym in sorted(expr.free_symbols, key=str):
        if sym in self.range_tree_nodes:
            replacements = {}
            for ps in self.range_tree_nodes[sym].precomputed_args():
                replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
            if len(replacements) > 0:
                self.range_tree_nodes[sym].expr = sympy_subs(self.range_tree_nodes[sym].expr, replacements)
            self.range_tree_nodes[sym].codegen()
    return expr