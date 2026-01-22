from os.path import abspath, dirname
from os.path import join as pjoin
import numpy as np
from nibabel.cmdline.diff import are_values_different
def test_diff_values_mixed():
    assert are_values_different(1.0, 1)
    assert are_values_different(1.0, '1')
    assert are_values_different(1, '1')
    assert are_values_different(1, None)
    assert are_values_different(np.ndarray([0]), 'hey')
    assert not are_values_different(None, None)