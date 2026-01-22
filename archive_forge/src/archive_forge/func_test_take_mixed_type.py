import pytest
import pandas._testing as tm
def test_take_mixed_type(self, float_string_frame):
    order = [4, 1, 2, 0, 3]
    for df in [float_string_frame]:
        result = df.take(order, axis=0)
        expected = df.reindex(df.index.take(order))
        tm.assert_frame_equal(result, expected)
        result = df.take(order, axis=1)
        expected = df.loc[:, ['foo', 'B', 'C', 'A', 'D']]
        tm.assert_frame_equal(result, expected)
    order = [4, 1, -2]
    for df in [float_string_frame]:
        result = df.take(order, axis=0)
        expected = df.reindex(df.index.take(order))
        tm.assert_frame_equal(result, expected)
        result = df.take(order, axis=1)
        expected = df.loc[:, ['foo', 'B', 'D']]
        tm.assert_frame_equal(result, expected)