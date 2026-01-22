import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test__row_tensor_product():
    dm1 = np.arange(1, 17).reshape((4, 4))
    assert np.array_equal(_row_tensor_product([dm1]), dm1)
    ones = np.ones(4).reshape((4, 1))
    tp1 = _row_tensor_product([ones, dm1])
    assert np.array_equal(tp1, dm1)
    tp2 = _row_tensor_product([dm1, ones])
    assert np.array_equal(tp2, dm1)
    twos = 2 * ones
    tp3 = _row_tensor_product([twos, dm1])
    assert np.array_equal(tp3, 2 * dm1)
    tp4 = _row_tensor_product([dm1, twos])
    assert np.array_equal(tp4, 2 * dm1)
    dm2 = np.array([[1, 2], [1, 2]])
    dm3 = np.arange(1, 7).reshape((2, 3))
    expected_tp5 = np.array([[1, 2, 3, 2, 4, 6], [4, 5, 6, 8, 10, 12]])
    tp5 = _row_tensor_product([dm2, dm3])
    assert np.array_equal(tp5, expected_tp5)
    expected_tp6 = np.array([[1, 2, 2, 4, 3, 6], [4, 8, 5, 10, 6, 12]])
    tp6 = _row_tensor_product([dm3, dm2])
    assert np.array_equal(tp6, expected_tp6)