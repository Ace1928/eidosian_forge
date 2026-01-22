import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def lower_yield(self, inst):
    yp = self.generator_info.yield_points[inst.index]
    assert yp.inst is inst
    self.genlower.init_generator_state(self)
    y = generators.LowerYield(self, yp, yp.live_vars | yp.weak_live_vars)
    y.lower_yield_suspend()
    val = self.loadvar(inst.value.name)
    self.pyapi.incref(val)
    self.call_conv.return_value(self.builder, val)
    y.lower_yield_resume()
    return self.pyapi.make_none()