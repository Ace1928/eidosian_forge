import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_datetime_frame_shift_with_freq(self, datetime_frame, frame_or_series):
    dtobj = tm.get_obj(datetime_frame, frame_or_series)
    shifted = dtobj.shift(1, freq='infer')
    unshifted = shifted.shift(-1, freq='infer')
    tm.assert_equal(dtobj, unshifted)
    shifted2 = dtobj.shift(freq=dtobj.index.freq)
    tm.assert_equal(shifted, shifted2)
    inferred_ts = DataFrame(datetime_frame.values, Index(np.asarray(datetime_frame.index)), columns=datetime_frame.columns)
    inferred_ts = tm.get_obj(inferred_ts, frame_or_series)
    shifted = inferred_ts.shift(1, freq='infer')
    expected = dtobj.shift(1, freq='infer')
    expected.index = expected.index._with_freq(None)
    tm.assert_equal(shifted, expected)
    unshifted = shifted.shift(-1, freq='infer')
    tm.assert_equal(unshifted, inferred_ts)