import copy
import numpy as np
from llvmlite import ir as lir
from numba.core import types, typing, utils, ir, config, ir_utils, registry
from numba.core.typing.templates import (CallableTemplate, signature,
from numba.core.imputils import lower_builtin
from numba.core.extending import register_jitable
from numba.core.errors import NumbaValueError
from numba.misc.special import literal_unroll
import numba
import operator
from numba.np import numpy_support
def compile_for_argtys(self, argtys, kwtys, return_type, sigret):
    _, result, typemap, calltypes = self._type_cache[argtys]
    new_func = self._stencil_wrapper(result, sigret, return_type, typemap, calltypes, *argtys)
    return new_func