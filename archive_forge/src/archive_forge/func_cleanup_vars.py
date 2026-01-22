import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def cleanup_vars(self):
    """
        Cleanup live variables.
        """
    for name in self._live_vars:
        ptr = self._getvar(name)
        self.decref(self.builder.load(ptr))