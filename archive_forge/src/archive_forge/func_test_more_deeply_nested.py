import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_more_deeply_nested(self, deep_nested):
    result = json_normalize(deep_nested, ['states', 'cities'], meta=['country', ['states', 'name']])
    ex_data = {'country': ['USA'] * 4 + ['Germany'] * 3, 'states.name': ['California', 'California', 'Ohio', 'Ohio', 'Bayern', 'Nordrhein-Westfalen', 'Nordrhein-Westfalen'], 'name': ['San Francisco', 'Los Angeles', 'Columbus', 'Cleveland', 'Munich', 'Duesseldorf', 'Koeln'], 'pop': [12345, 12346, 1234, 1236, 12347, 1238, 1239]}
    expected = DataFrame(ex_data, columns=result.columns)
    tm.assert_frame_equal(result, expected)