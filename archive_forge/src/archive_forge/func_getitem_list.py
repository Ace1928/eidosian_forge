import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@lower_builtin(operator.getitem, types.List, types.Integer)
def getitem_list(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    index = args[1]
    index = inst.fix_index(index)
    inst.guard_index(index, msg='getitem out of range')
    result = inst.getitem(index)
    return impl_ret_borrowed(context, builder, sig.return_type, result)