import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def assert_level_values(frame, levels, axis):
    axis = frame._get_axis_number(axis)
    for level in levels:
        if axis == 0:
            expected = frame.index.get_level_values(level=level)._values
        else:
            expected = frame.columns.get_level_values(level=level)._values
        result = frame._get_label_or_level_values(level, axis=axis)
        assert array_equivalent(expected, result)