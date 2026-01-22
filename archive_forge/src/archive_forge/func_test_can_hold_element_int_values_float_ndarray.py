import numpy as np
from pandas.core.dtypes.cast import can_hold_element
def test_can_hold_element_int_values_float_ndarray():
    arr = np.array([], dtype=np.int64)
    element = np.array([1.0, 2.0])
    assert can_hold_element(arr, element)
    assert not can_hold_element(arr, element + 0.5)
    element = np.array([3, 2 ** 65], dtype=np.float64)
    assert not can_hold_element(arr, element)