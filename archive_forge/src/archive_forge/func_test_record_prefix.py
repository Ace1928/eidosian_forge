import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_record_prefix(self, state_data):
    result = json_normalize(state_data[0], 'counties')
    expected = DataFrame(state_data[0]['counties'])
    tm.assert_frame_equal(result, expected)
    result = json_normalize(state_data, 'counties', meta='state', record_prefix='county_')
    expected = []
    for rec in state_data:
        expected.extend(rec['counties'])
    expected = DataFrame(expected)
    expected = expected.rename(columns=lambda x: 'county_' + x)
    expected['state'] = np.array(['Florida', 'Ohio']).repeat([3, 2])
    tm.assert_frame_equal(result, expected)