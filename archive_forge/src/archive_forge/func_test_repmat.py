import numpy as np
import numpy.matlib
from numpy.testing import assert_array_equal, assert_
def test_repmat():
    a1 = np.arange(4)
    x = numpy.matlib.repmat(a1, 2, 2)
    y = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]])
    assert_array_equal(x, y)