from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_translate_mixed_object():
    s = Series(['a', 'b', 'c', 1.2])
    table = str.maketrans('abc', 'cde')
    expected = Series(['c', 'd', 'e', np.nan], dtype=object)
    result = s.str.translate(table)
    tm.assert_series_equal(result, expected)