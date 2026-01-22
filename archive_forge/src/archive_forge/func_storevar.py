import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def storevar(self, value, name, clobber=False):
    """
        Stores a llvm value and allocate stack slot if necessary.
        The llvm value can be of arbitrary type.
        """
    is_redefine = name in self._live_vars and (not clobber)
    ptr = self._getvar(name, ltype=value.type)
    if is_redefine:
        old = self.builder.load(ptr)
    else:
        self._live_vars.add(name)
    assert value.type == ptr.type.pointee, (str(value.type), str(ptr.type.pointee))
    self.builder.store(value, ptr)
    if is_redefine:
        self.decref(old)