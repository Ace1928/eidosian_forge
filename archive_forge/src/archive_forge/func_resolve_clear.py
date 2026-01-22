import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@bound_function('list.clear')
def resolve_clear(self, list, args, kws):
    assert not args
    assert not kws
    return signature(types.none)