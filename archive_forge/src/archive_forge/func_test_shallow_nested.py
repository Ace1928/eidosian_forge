import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_shallow_nested(self):
    data = [{'state': 'Florida', 'shortname': 'FL', 'info': {'governor': 'Rick Scott'}, 'counties': [{'name': 'Dade', 'population': 12345}, {'name': 'Broward', 'population': 40000}, {'name': 'Palm Beach', 'population': 60000}]}, {'state': 'Ohio', 'shortname': 'OH', 'info': {'governor': 'John Kasich'}, 'counties': [{'name': 'Summit', 'population': 1234}, {'name': 'Cuyahoga', 'population': 1337}]}]
    result = json_normalize(data, 'counties', ['state', 'shortname', ['info', 'governor']])
    ex_data = {'name': ['Dade', 'Broward', 'Palm Beach', 'Summit', 'Cuyahoga'], 'state': ['Florida'] * 3 + ['Ohio'] * 2, 'shortname': ['FL', 'FL', 'FL', 'OH', 'OH'], 'info.governor': ['Rick Scott'] * 3 + ['John Kasich'] * 2, 'population': [12345, 40000, 60000, 1234, 1337]}
    expected = DataFrame(ex_data, columns=result.columns)
    tm.assert_frame_equal(result, expected)