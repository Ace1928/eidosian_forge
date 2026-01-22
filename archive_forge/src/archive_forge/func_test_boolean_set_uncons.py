from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_boolean_set_uncons(self, float_frame):
    float_frame['E'] = 7.0
    expected = float_frame.values.copy()
    expected[expected > 1] = 2
    float_frame[float_frame > 1] = 2
    tm.assert_almost_equal(expected, float_frame.values)