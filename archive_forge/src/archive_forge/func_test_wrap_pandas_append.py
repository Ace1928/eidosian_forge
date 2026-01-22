from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_wrap_pandas_append():
    a = gen_data(1, True)
    a.name = 'apple'
    b = gen_data(1, False)
    wrapped = PandasWrapper(a).wrap(b, append='appended')
    expected = 'apple_appended'
    assert wrapped.name == expected
    a = gen_data(2, True)
    a.columns = ['apple_' + str(i) for i in range(a.shape[1])]
    b = gen_data(2, False)
    wrapped = PandasWrapper(a).wrap(b, append='appended')
    expected = [c + '_appended' for c in a.columns]
    assert list(wrapped.columns) == expected