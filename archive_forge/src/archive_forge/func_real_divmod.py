import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def real_divmod(context, builder, x, y):
    assert x.type == y.type
    floatty = x.type
    module = builder.module
    fname = context.mangler('.numba.python.rem', [x.type])
    fnty = ir.FunctionType(floatty, (floatty, floatty, ir.PointerType(floatty)))
    fn = cgutils.get_or_insert_function(module, fnty, fname)
    if fn.is_declaration:
        fn.linkage = 'linkonce_odr'
        fnbuilder = ir.IRBuilder(fn.append_basic_block('entry'))
        fx, fy, pmod = fn.args
        div, mod = real_divmod_func_body(context, fnbuilder, fx, fy)
        fnbuilder.store(mod, pmod)
        fnbuilder.ret(div)
    pmod = cgutils.alloca_once(builder, floatty)
    quotient = builder.call(fn, (x, y, pmod))
    return (quotient, builder.load(pmod))