import numpy
import numpy.linalg as NLA
import pytest
import modin.numpy as np
import modin.numpy.linalg as LA
import modin.pandas as pd
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('axis', [None, 0, 1], ids=['axis=None', 'axis=0', 'axis=1'])
def test_norm_fro_2d(axis):
    x1 = numpy.random.randint(-10, 10, size=(100, 3))
    numpy_result = NLA.norm(x1, axis=axis)
    x1 = np.array(x1)
    modin_result = LA.norm(x1, axis=axis)
    if isinstance(modin_result, np.array):
        modin_result = modin_result._to_numpy()
    numpy.testing.assert_allclose(modin_result, numpy_result, rtol=1e-12)