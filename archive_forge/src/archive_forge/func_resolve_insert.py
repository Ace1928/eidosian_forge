import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@bound_function('list.insert')
def resolve_insert(self, list, args, kws):
    idx, item = args
    assert not kws
    if isinstance(idx, types.Integer):
        unified = self.context.unify_pairs(list.dtype, item)
        if unified is not None:
            sig = signature(types.none, types.intp, unified)
            sig = sig.replace(recvr=list.copy(dtype=unified))
            return sig