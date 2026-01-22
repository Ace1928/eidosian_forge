import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def ufunc_find_matching_loop(ufunc, arg_types):
    """Find the appropriate loop to be used for a ufunc based on the types
    of the operands

    ufunc        - The ufunc we want to check
    arg_types    - The tuple of arguments to the ufunc, including any
                   explicit output(s).
    return value - A UFuncLoopSpec identifying the loop, or None
                   if no matching loop is found.
    """
    input_types = arg_types[:ufunc.nin]
    output_types = arg_types[ufunc.nin:]
    assert len(input_types) == ufunc.nin
    try:
        np_input_types = [as_dtype(x) for x in input_types]
    except errors.NumbaNotImplementedError:
        return None
    try:
        np_output_types = [as_dtype(x) for x in output_types]
    except errors.NumbaNotImplementedError:
        return None
    has_mixed_inputs = any((dt.kind in 'iu' for dt in np_input_types)) and any((dt.kind in 'cf' for dt in np_input_types))

    def choose_types(numba_types, ufunc_letters):
        """
        Return a list of Numba types representing *ufunc_letters*,
        except when the letter designates a datetime64 or timedelta64,
        in which case the type is taken from *numba_types*.
        """
        assert len(ufunc_letters) >= len(numba_types)
        types = [tp if letter in 'mM' else from_dtype(np.dtype(letter)) for tp, letter in zip(numba_types, ufunc_letters)]
        types += [from_dtype(np.dtype(letter)) for letter in ufunc_letters[len(numba_types):]]
        return types

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
    for candidate in ufunc.types:
        ufunc_inputs = candidate[:ufunc.nin]
        ufunc_outputs = candidate[-ufunc.nout:] if ufunc.nout else []
        if 'e' in ufunc_inputs:
            continue
        if 'O' in ufunc_inputs:
            continue
        found = True
        for outer, inner in zip(np_input_types, ufunc_inputs):
            if outer.char in 'mM' or inner in 'mM':
                if outer.char != inner:
                    found = False
                    break
            elif not ufunc_can_cast(outer.char, inner, has_mixed_inputs, 'safe'):
                found = False
                break
        if found:
            for outer, inner in zip(np_output_types, ufunc_outputs):
                if outer.char not in 'mM' and (not ufunc_can_cast(inner, outer.char, has_mixed_inputs, 'same_kind')):
                    found = False
                    break
        if found:
            try:
                inputs = choose_types(input_types, ufunc_inputs)
                outputs = choose_types(output_types, ufunc_outputs)
                if ufunc_inputs[0] == 'm' or ufunc_inputs == 'Mm':
                    outputs = set_output_dt_units(inputs, outputs, ufunc_inputs, ufunc.__name__)
            except errors.NumbaNotImplementedError:
                continue
            else:
                return UFuncLoopSpec(inputs, outputs, candidate)
    return None