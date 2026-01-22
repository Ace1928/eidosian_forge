from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_frame_float(self, float_frame):
    piece = float_frame.loc[float_frame.index[:2], ['A', 'B']]
    float_frame.loc[float_frame.index[-2]:, ['A', 'B']] = piece.values
    result = float_frame.loc[float_frame.index[-2:], ['A', 'B']].values
    expected = piece.values
    tm.assert_almost_equal(result, expected)