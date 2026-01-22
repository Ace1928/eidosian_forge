import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_flat_stays_flat(self):
    recs = [{'flat1': 1, 'flat2': 2}, {'flat3': 3, 'flat2': 4}]
    result = nested_to_record(recs)
    expected = recs
    assert result == expected