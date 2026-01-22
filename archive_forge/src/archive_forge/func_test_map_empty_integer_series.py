from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_empty_integer_series():
    s = Series([], dtype=int)
    result = s.map(lambda x: x)
    tm.assert_series_equal(result, s)