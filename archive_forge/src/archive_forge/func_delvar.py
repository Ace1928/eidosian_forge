import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def delvar(self, name):
    """
        Delete the variable slot with the given name. This will decref
        the corresponding Python object.
        """
    self._live_vars.remove(name)
    ptr = self._getvar(name)
    self.decref(self.builder.load(ptr))
    self.builder.store(cgutils.get_null_value(ptr.type.pointee), ptr)