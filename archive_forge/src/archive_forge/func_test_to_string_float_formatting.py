from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_float_formatting(self):
    with option_context('display.precision', 5, 'display.notebook_repr_html', False):
        df = DataFrame({'x': [0, 0.25, 3456.0, 1.2e+46, 1640000.0, 170000000.0, 1.253456, np.pi, -1000000.0]})
        df_s = df.to_string()
        if _three_digit_exp():
            expected = '              x\n0  0.00000e+000\n1  2.50000e-001\n2  3.45600e+003\n3  1.20000e+046\n4  1.64000e+006\n5  1.70000e+008\n6  1.25346e+000\n7  3.14159e+000\n8 -1.00000e+006'
        else:
            expected = '             x\n0  0.00000e+00\n1  2.50000e-01\n2  3.45600e+03\n3  1.20000e+46\n4  1.64000e+06\n5  1.70000e+08\n6  1.25346e+00\n7  3.14159e+00\n8 -1.00000e+06'
        assert df_s == expected
        df = DataFrame({'x': [3234, 0.253]})
        df_s = df.to_string()
        expected = '          x\n0  3234.000\n1     0.253'
        assert df_s == expected
    assert get_option('display.precision') == 6
    df = DataFrame({'x': [1000000000.0, 0.2512]})
    df_s = df.to_string()
    if _three_digit_exp():
        expected = '               x\n0  1.000000e+009\n1  2.512000e-001'
    else:
        expected = '              x\n0  1.000000e+09\n1  2.512000e-01'
    assert df_s == expected