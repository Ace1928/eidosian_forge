import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def typeof_assign(self, inst):
    value = inst.value
    if isinstance(value, ir.Const):
        self.typeof_const(inst, inst.target, value.value)
    elif isinstance(value, ir.Var):
        self.constraints.append(Propagate(dst=inst.target.name, src=value.name, loc=inst.loc))
    elif isinstance(value, (ir.Global, ir.FreeVar)):
        self.typeof_global(inst, inst.target, value)
    elif isinstance(value, ir.Arg):
        self.typeof_arg(inst, inst.target, value)
    elif isinstance(value, ir.Expr):
        self.typeof_expr(inst, inst.target, value)
    elif isinstance(value, ir.Yield):
        self.typeof_yield(inst, inst.target, value)
    else:
        msg = 'Unsupported assignment encountered: %s %s' % (type(value), str(value))
        raise UnsupportedError(msg, loc=inst.loc)