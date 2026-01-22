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
def indexing(self, index: sympy.Expr, *, copy_shape=None, dense_indexing=False, override_mask=None):
    """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
    index = self.simplify_indexing(index)
    index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
    if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
        index = index.subs(V.graph.sizevars.precomputed_replacements)
    if len(index.atoms(sympy.ceiling)):
        for a in index.atoms(sympy.ceiling):
            symbols = a.free_symbols
            if len(symbols) > 0 and all((s.name.startswith('s') or s.name.startswith('ps') for s in symbols)):
                replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                index = sympy_subs(index, replacements)
    index_vars = index.free_symbols
    index = self.simplify_indexing(index)
    index_str = self.index_to_str(index)
    mask_vars: Set[str] = set()
    for var in index_vars:
        assert isinstance(var, sympy.Symbol)
        if override_mask:
            pass
        elif var.name.startswith('tmp'):
            cse_var = self.cse.varname_map[var.name]
            mask_vars.update(cse_var.mask_vars)
        elif var.name.startswith(('s', 'ps', 'i')):
            pass
        else:
            assert var.name[0] in 'xyr', var.name
            mask_vars.add(f'{var.name[0]}mask')
    need_dense = (config.triton.dense_indexing or dense_indexing or self._load_mask is not None) and index != 0
    have_dense = True
    have_loop_vars = False
    dense_mask_vars = set()
    for tree in self.range_trees:
        if tree.prefix == 'r' and (not self.inside_reduction):
            continue
        if index_vars.intersection(tree.var_list):
            have_loop_vars = True
        else:
            have_dense = False
        dense_mask_vars.add(f'{tree.prefix}mask')
    expand_str = None
    if isinstance(index, sympy.Integer):
        expand_str = f'{copy_shape}.shape' if copy_shape else self.dense_size_str()
        index_str = f'tl.full({expand_str}, {index_str}, tl.int32)'
        return (index_str, set(), 'None', expand_str)
    if need_dense and (not have_dense):
        expand_str = f'{copy_shape}.shape' if copy_shape else self.dense_size_str()
        index_str = f'tl.broadcast_to({index_str}, {expand_str})'
        mask_vars = dense_mask_vars
    elif not have_loop_vars and copy_shape:
        index_str = f'tl.broadcast_to({index_str}, {copy_shape}.shape)'
        mask_vars = dense_mask_vars
    if override_mask:
        mask_vars = {override_mask}
    if self._load_mask:
        mask_vars.add(self._load_mask)
    self.filter_masks(mask_vars)
    mask_str = ' & '.join(sorted(map(str, mask_vars))) if mask_vars else 'None'
    return (index_str, mask_vars, mask_str, expand_str)