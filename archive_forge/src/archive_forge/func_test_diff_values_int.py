from os.path import abspath, dirname
from os.path import join as pjoin
import numpy as np
from nibabel.cmdline.diff import are_values_different
def test_diff_values_int():
    large = 10 ** 30
    assert not are_values_different(0, 0)
    assert not are_values_different(1, 1)
    assert not are_values_different(large, large)
    assert are_values_different(0, 1)
    assert are_values_different(1, 2)
    assert are_values_different(1, large)