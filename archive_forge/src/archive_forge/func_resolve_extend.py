import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@bound_function('list.extend')
def resolve_extend(self, list, args, kws):
    iterable, = args
    assert not kws
    if not isinstance(iterable, types.IterableType):
        return
    dtype = iterable.iterator_type.yield_type
    unified = self.context.unify_pairs(list.dtype, dtype)
    if unified is not None:
        sig = signature(types.none, iterable)
        sig = sig.replace(recvr=list.copy(dtype=unified))
        return sig