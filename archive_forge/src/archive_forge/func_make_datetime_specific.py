import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def make_datetime_specific(outputs, dt_unit, td_unit):
    new_outputs = []
    for out in outputs:
        if isinstance(out, types.NPDatetime) and out.unit == '':
            unit = npdatetime_helpers.combine_datetime_timedelta_units(dt_unit, td_unit)
            if unit is None:
                raise TypeError(f"ufunc '{ufunc_name}' is not " + 'supported between ' + f'datetime64[{dt_unit}] ' + f'and timedelta64[{td_unit}]')
            new_outputs.append(types.NPDatetime(unit))
        else:
            new_outputs.append(out)
    return new_outputs