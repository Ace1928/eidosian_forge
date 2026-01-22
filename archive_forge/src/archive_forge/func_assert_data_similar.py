from io import BytesIO
import numpy as np
from ..optpkg import optional_package
from numpy.testing import assert_array_equal
def assert_data_similar(arr, params):
    """Check data is the same if recorded, otherwise check summaries

    Helper function to test image array data `arr` against record in `params`,
    where record can be the array itself, or summary values from the array.

    Parameters
    ----------
    arr : array-like
        Something that results in an array after ``np.asarry(arr)``
    params : mapping
        Mapping that has either key ``data`` with value that is array-like, or
        key ``data_summary`` with value a dict having keys ``min``, ``max``,
        ``mean``
    """
    if 'data' in params:
        assert_array_equal(arr, params['data'])
        return
    summary = params['data_summary']
    real_arr = np.asarray(arr)
    assert np.allclose((real_arr.min(), real_arr.max(), real_arr.mean()), (summary['min'], summary['max'], summary['mean']))