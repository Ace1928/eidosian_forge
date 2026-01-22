import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def set_output_dt_units(inputs, outputs, ufunc_inputs, ufunc_name):
    """
        Sets the output unit of a datetime type based on the input units

        Timedelta is a special dtype that requires the time unit to be
        specified (day, month, etc). Not every operation with timedelta inputs
        leads to an output of timedelta output. However, for those that do,
        the unit of output must be inferred based on the units of the inputs.

        At the moment this function takes care of two cases:
        a) where all inputs are timedelta with the same unit (mm), and
        therefore the output has the same unit.
        This case is used for arr.sum, and for arr1+arr2 where all arrays
        are timedeltas.
        If in the future this needs to be extended to a case with mixed units,
        the rules should be implemented in `npdatetime_helpers` and called
        from this function to set the correct output unit.
        b) where left operand is a timedelta, i.e. the "m?" case. This case
        is used for division, eg timedelta / int.

        At the time of writing, Numba does not support addition of timedelta
        and other types, so this function does not consider the case "?m",
        i.e. where timedelta is the right operand to a non-timedelta left
        operand. To extend it in the future, just add another elif clause.
        """

    def make_specific(outputs, unit):
        new_outputs = []
        for out in outputs:
            if isinstance(out, types.NPTimedelta) and out.unit == '':
                new_outputs.append(types.NPTimedelta(unit))
            else:
                new_outputs.append(out)
        return new_outputs

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
    if ufunc_inputs == 'mm':
        if all((inp.unit == inputs[0].unit for inp in inputs)):
            unit = inputs[0].unit
            new_outputs = make_specific(outputs, unit)
        else:
            return outputs
        return new_outputs
    elif ufunc_inputs == 'mM':
        td_unit = inputs[0].unit
        dt_unit = inputs[1].unit
        return make_datetime_specific(outputs, dt_unit, td_unit)
    elif ufunc_inputs == 'Mm':
        dt_unit = inputs[0].unit
        td_unit = inputs[1].unit
        return make_datetime_specific(outputs, dt_unit, td_unit)
    elif ufunc_inputs[0] == 'm':
        unit = inputs[0].unit
        new_outputs = make_specific(outputs, unit)
        return new_outputs