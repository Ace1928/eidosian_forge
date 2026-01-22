import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_normalize_with_multichar_separator(self):
    data = {'a': [1, 2], 'b': {'b_1': 2, 'b_2': (3, 4)}}
    result = json_normalize(data, sep='__')
    expected = DataFrame([[[1, 2], 2, (3, 4)]], columns=['a', 'b__b_1', 'b__b_2'])
    tm.assert_frame_equal(result, expected)