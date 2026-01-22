import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_value_array_record_prefix(self):
    result = json_normalize({'A': [1, 2]}, 'A', record_prefix='Prefix.')
    expected = DataFrame([[1], [2]], columns=['Prefix.0'])
    tm.assert_frame_equal(result, expected)