import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_by_zero(self, datetime_frame, frame_or_series):
    obj = tm.get_obj(datetime_frame, frame_or_series)
    unshifted = obj.shift(0)
    tm.assert_equal(unshifted, obj)