import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def rename_labels(blocks):
    """rename labels of function body blocks according to topological sort.
    The set of labels of these blocks will remain unchanged.
    """
    topo_order = find_topo_order(blocks)
    return_label = -1
    for l, b in blocks.items():
        if isinstance(b.body[-1], ir.Return):
            return_label = l
    if return_label != -1:
        topo_order.remove(return_label)
        topo_order.append(return_label)
    label_map = {}
    all_labels = sorted(topo_order, reverse=True)
    for label in topo_order:
        label_map[label] = all_labels.pop()
    for b in blocks.values():
        term = b.terminator
        if isinstance(term, ir.Jump):
            term.target = label_map[term.target]
        if isinstance(term, ir.Branch):
            term.truebr = label_map[term.truebr]
            term.falsebr = label_map[term.falsebr]
    new_blocks = {}
    for k, b in blocks.items():
        new_label = label_map[k]
        new_blocks[new_label] = b
    return new_blocks