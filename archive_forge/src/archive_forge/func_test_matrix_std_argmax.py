import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_matrix_std_argmax(self):
    x = np.asmatrix(np.random.uniform(0, 1, (3, 3)))
    assert_equal(x.std().shape, ())
    assert_equal(x.argmax().shape, ())