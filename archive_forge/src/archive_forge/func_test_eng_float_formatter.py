import numpy as np
import pytest
from pandas import (
from pandas.io.formats.format import EngFormatter
def test_eng_float_formatter(self):
    df = DataFrame({'A': [1.41, 141.0, 14100, 1410000.0]})
    set_eng_float_format()
    result = df.to_string()
    expected = '             A\n0    1.410E+00\n1  141.000E+00\n2   14.100E+03\n3    1.410E+06'
    assert result == expected
    set_eng_float_format(use_eng_prefix=True)
    result = df.to_string()
    expected = '         A\n0    1.410\n1  141.000\n2  14.100k\n3   1.410M'
    assert result == expected
    set_eng_float_format(accuracy=0)
    result = df.to_string()
    expected = '         A\n0    1E+00\n1  141E+00\n2   14E+03\n3    1E+06'
    assert result == expected