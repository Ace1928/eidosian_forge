from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method', ['partition', 'rpartition'])
def test_partition_sep_kwarg(any_string_dtype, method):
    s = Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'], dtype=any_string_dtype)
    expected = getattr(s.str, method)(sep='_')
    result = getattr(s.str, method)('_')
    tm.assert_frame_equal(result, expected)