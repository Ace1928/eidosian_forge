import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def lower_const(self, const):
    index = self.env_manager.add_const(const)
    ret = self.env_manager.read_const(index)
    self.check_error(ret)
    self.incref(ret)
    return ret