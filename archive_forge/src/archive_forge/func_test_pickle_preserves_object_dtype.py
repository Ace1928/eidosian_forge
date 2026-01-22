from pandas import Index
import pandas._testing as tm
def test_pickle_preserves_object_dtype():
    index = Index([1, 2, 3], dtype=object)
    result = tm.round_trip_pickle(index)
    assert result.dtype == object
    tm.assert_index_equal(index, result)