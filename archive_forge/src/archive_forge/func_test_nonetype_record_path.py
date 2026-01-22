import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_nonetype_record_path(self, nulls_fixture):
    result = json_normalize([{'state': 'Texas', 'info': nulls_fixture}, {'state': 'Florida', 'info': [{'i': 2}]}], record_path=['info'])
    expected = DataFrame({'i': 2}, index=[0])
    tm.assert_equal(result, expected)