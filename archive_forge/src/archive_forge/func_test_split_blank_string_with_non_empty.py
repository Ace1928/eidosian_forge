from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_blank_string_with_non_empty(any_string_dtype):
    values = Series(['a b c', 'a b', '', ' '], name='test', dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame([['a', 'b', 'c'], ['a', 'b', None], [None, None, None], [None, None, None]], dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)