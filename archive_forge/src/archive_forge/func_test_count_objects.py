from pandas import (
import pandas._testing as tm
def test_count_objects(self, float_string_frame):
    dm = DataFrame(float_string_frame._series)
    df = DataFrame(float_string_frame._series)
    tm.assert_series_equal(dm.count(), df.count())
    tm.assert_series_equal(dm.count(1), df.count(1))