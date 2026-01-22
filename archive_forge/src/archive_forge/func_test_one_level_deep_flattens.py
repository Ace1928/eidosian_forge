import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_one_level_deep_flattens(self):
    data = {'flat1': 1, 'dict1': {'c': 1, 'd': 2}}
    result = nested_to_record(data)
    expected = {'dict1.c': 1, 'dict1.d': 2, 'flat1': 1}
    assert result == expected