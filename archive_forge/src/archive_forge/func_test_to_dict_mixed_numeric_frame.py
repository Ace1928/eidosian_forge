from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_mixed_numeric_frame(self):
    df = DataFrame({'a': [1.0], 'b': [9.0]})
    result = df.reset_index().to_dict('records')
    expected = [{'index': 0, 'a': 1.0, 'b': 9.0}]
    assert result == expected