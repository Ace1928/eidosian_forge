import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('data_out', [numpy.zeros((1, 3)), numpy.zeros((2, 3))])
def test_out_broadcast(data_out):
    if data_out.shape == (2, 3):
        pytest.xfail('broadcasting would require duplicating row: see GH#5819')
    data1 = [[1, 2, 3]]
    data2 = [7, 8, 9]
    modin_out, numpy_out = (np.array(data_out), numpy.array(data_out))
    numpy.add(numpy.array(data1), numpy.array(data2), out=numpy_out)
    np.add(np.array(data1), np.array(data2), out=modin_out)
    assert_scalar_or_array_equal(modin_out, numpy_out)