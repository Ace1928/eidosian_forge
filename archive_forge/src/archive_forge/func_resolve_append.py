import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@bound_function('list.append')
def resolve_append(self, list, args, kws):
    item, = args
    assert not kws
    unified = self.context.unify_pairs(list.dtype, item)
    if unified is not None:
        sig = signature(types.none, unified)
        sig = sig.replace(recvr=list.copy(dtype=unified))
        return sig