from os.path import abspath, dirname
from os.path import join as pjoin
import numpy as np
from nibabel.cmdline.diff import are_values_different
def test_diff_values_array():
    from numpy import array, inf, nan
    a_int = array([1, 2])
    a_float = a_int.astype(float)
    assert are_values_different(a_int, a_float)
    assert are_values_different(a_int, a_int, a_float)
    assert are_values_different(np.arange(3), np.arange(1, 4))
    assert are_values_different(np.arange(3), np.arange(4))
    assert are_values_different(np.arange(4), np.arange(4).reshape((2, 2)))
    assert are_values_different(array([1]), array([1, 1]))
    assert not are_values_different(a_int, a_int)
    assert not are_values_different(a_float, a_float)
    assert not are_values_different(nan, nan)
    assert not are_values_different(nan, nan, nan)
    assert are_values_different(nan, nan, 1)
    assert are_values_different(1, nan, nan)
    assert not are_values_different(array([nan, nan]), array([nan, nan]))
    assert not are_values_different(array([nan, nan]), array([nan, nan]), array([nan, nan]))
    assert not are_values_different(array([nan, 1]), array([nan, 1]))
    assert are_values_different(array([nan, nan]), array([nan, 1]))
    assert are_values_different(array([0, nan]), array([nan, 0]))
    assert are_values_different(array([1, 2, 3, nan]), array([nan, 3, 5, 4]))
    assert are_values_different(nan, 1.0)
    assert are_values_different(array([1, 2, 3, nan]), array([3, 4, 5, nan]))
    assert not are_values_different(array([0, inf]), array([0, inf]))
    assert are_values_different(array([0, inf]), array([inf, 0]))
    assert not are_values_different(np.array(1, dtype='<i4'), np.array(1, dtype='>i4'))
    assert are_values_different(np.array(1, dtype='<i4'), np.array(1, dtype='<i2'))