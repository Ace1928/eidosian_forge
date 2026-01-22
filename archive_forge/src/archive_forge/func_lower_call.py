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
def lower_call(self, resty, expr):
    signature = self.fndesc.calltypes[expr]
    self.debug_print('# lower_call: expr = {0}'.format(expr))
    if isinstance(signature.return_type, types.Phantom):
        return self.context.get_dummy_value()
    fnty = self.typeof(expr.func.name)
    if isinstance(fnty, types.ObjModeDispatcher):
        res = self._lower_call_ObjModeDispatcher(fnty, expr, signature)
    elif isinstance(fnty, types.ExternalFunction):
        res = self._lower_call_ExternalFunction(fnty, expr, signature)
    elif isinstance(fnty, types.ExternalFunctionPointer):
        res = self._lower_call_ExternalFunctionPointer(fnty, expr, signature)
    elif isinstance(fnty, types.RecursiveCall):
        res = self._lower_call_RecursiveCall(fnty, expr, signature)
    elif isinstance(fnty, types.FunctionType):
        res = self._lower_call_FunctionType(fnty, expr, signature)
    else:
        res = self._lower_call_normal(fnty, expr, signature)
    if res is None:
        if signature.return_type == types.void:
            res = self.context.get_dummy_value()
        else:
            raise LoweringError(msg='non-void function returns None from implementation', loc=self.loc)
    return self.context.cast(self.builder, res, signature.return_type, resty)