import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def categorical_to_int(data, levels, NA_action, origin=None):
    assert isinstance(levels, tuple)
    if safe_is_pandas_categorical(data):
        data_levels_tuple = tuple(pandas_Categorical_categories(data))
        if not data_levels_tuple == levels:
            raise PatsyError('mismatching levels: expected %r, got %r' % (levels, data_levels_tuple), origin)
        return pandas_Categorical_codes(data)
    if isinstance(data, _CategoricalBox):
        if data.levels is not None and tuple(data.levels) != levels:
            raise PatsyError('mismatching levels: expected %r, got %r' % (levels, tuple(data.levels)), origin)
        data = data.data
    data = _categorical_shape_fix(data)
    try:
        level_to_int = dict(zip(levels, range(len(levels))))
    except TypeError:
        raise PatsyError('Error interpreting categorical data: all items must be hashable', origin)
    if hasattr(data, 'dtype') and safe_issubdtype(data.dtype, np.bool_):
        if level_to_int[False] == 0 and level_to_int[True] == 1:
            return data.astype(np.int_)
    out = np.empty(len(data), dtype=int)
    for i, value in enumerate(data):
        if NA_action.is_categorical_NA(value):
            out[i] = -1
        else:
            try:
                out[i] = level_to_int[value]
            except KeyError:
                SHOW_LEVELS = 4
                level_strs = []
                if len(levels) <= SHOW_LEVELS:
                    level_strs += [repr(level) for level in levels]
                else:
                    level_strs += [repr(level) for level in levels[:SHOW_LEVELS // 2]]
                    level_strs.append('...')
                    level_strs += [repr(level) for level in levels[-SHOW_LEVELS // 2:]]
                level_str = '[%s]' % ', '.join(level_strs)
                raise PatsyError('Error converting data to categorical: observation with value %r does not match any of the expected levels (expected: %s)' % (value, level_str), origin)
            except TypeError:
                raise PatsyError('Error converting data to categorical: encountered unhashable value %r' % (value,), origin)
    if have_pandas and isinstance(data, pandas.Series):
        out = pandas.Series(out, index=data.index)
    return out