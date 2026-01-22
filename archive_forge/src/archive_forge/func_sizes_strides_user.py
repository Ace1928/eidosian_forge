import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def sizes_strides_user(sizes, strides):
    import sympy
    from torch.fx.experimental.symbolic_shapes import eval_is_non_overlapping_and_dense
    for a in itertools.chain(sizes, strides):
        if isinstance(a, SymInt):
            return wrap_node(getattr(a.node, method)([to_node(a.node, b) for b in sizes], [to_node(a.node, b) for b in strides]))
    if method == 'is_non_overlapping_and_dense_indicator':
        return eval_is_non_overlapping_and_dense(sizes, strides)
    else:
        return bool(func([sympy.sympify(a) for a in sizes], [sympy.sympify(a) for a in strides]))