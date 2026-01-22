import numpy as np
import operator
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core import types, cgutils
from numba.core.cgutils import create_constant_array
from numba.core.imputils import (lower_builtin, lower_constant,
from numba.np import npdatetime_helpers, numpy_support, npyfuncs
from numba.extending import overload_method
from numba.core.config import IS_32BITS
from numba.core.errors import LoweringError
def year_to_days(builder, year_val):
    """
    Given a year *year_val* (offset to 1970), return the number of days
    since the 1970 epoch.
    """
    ret = cgutils.alloca_once(builder, TIMEDELTA64)
    days = scale_by_constant(builder, year_val, 365)
    with builder.if_else(cgutils.is_neg_int(builder, year_val)) as (if_neg, if_pos):
        with if_pos:
            from_1968 = add_constant(builder, year_val, 1)
            p_days = builder.add(days, unscale_by_constant(builder, from_1968, 4))
            from_1900 = add_constant(builder, from_1968, 68)
            p_days = builder.sub(p_days, unscale_by_constant(builder, from_1900, 100))
            from_1600 = add_constant(builder, from_1900, 300)
            p_days = builder.add(p_days, unscale_by_constant(builder, from_1600, 400))
            builder.store(p_days, ret)
        with if_neg:
            from_1972 = add_constant(builder, year_val, -2)
            n_days = builder.add(days, unscale_by_constant(builder, from_1972, 4))
            from_2000 = add_constant(builder, from_1972, -28)
            n_days = builder.sub(n_days, unscale_by_constant(builder, from_2000, 100))
            n_days = builder.add(n_days, unscale_by_constant(builder, from_2000, 400))
            builder.store(n_days, ret)
    return builder.load(ret)