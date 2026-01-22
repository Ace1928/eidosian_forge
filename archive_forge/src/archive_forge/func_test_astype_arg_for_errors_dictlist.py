import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_arg_for_errors_dictlist(self):
    df = DataFrame([{'a': '1', 'b': '16.5%', 'c': 'test'}, {'a': '2.2', 'b': '15.3', 'c': 'another_test'}])
    expected = DataFrame([{'a': 1.0, 'b': '16.5%', 'c': 'test'}, {'a': 2.2, 'b': '15.3', 'c': 'another_test'}])
    expected['c'] = expected['c'].astype('object')
    type_dict = {'a': 'float64', 'b': 'float64', 'c': 'object'}
    result = df.astype(dtype=type_dict, errors='ignore')
    tm.assert_frame_equal(result, expected)