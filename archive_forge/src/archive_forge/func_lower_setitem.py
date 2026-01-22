from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def lower_setitem(self, target_var, index_var, value_var, signature):
    target = self.loadvar(target_var.name)
    value = self.loadvar(value_var.name)
    index = self.loadvar(index_var.name)
    targetty = self.typeof(target_var.name)
    valuety = self.typeof(value_var.name)
    indexty = self.typeof(index_var.name)
    op = operator.setitem
    fnop = self.context.typing_context.resolve_value_type(op)
    callsig = fnop.get_call_type(self.context.typing_context, signature.args, {})
    impl = self.context.get_function(fnop, callsig)
    if isinstance(targetty, types.Optional):
        target = self.context.cast(self.builder, target, targetty, targetty.type)
    else:
        ul = types.unliteral
        assert ul(targetty) == ul(signature.args[0])
    index = self.context.cast(self.builder, index, indexty, signature.args[1])
    value = self.context.cast(self.builder, value, valuety, signature.args[2])
    return impl(self.builder, (target, index, value))