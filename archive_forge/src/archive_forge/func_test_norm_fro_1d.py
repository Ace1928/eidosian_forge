import numpy
import numpy.linalg as NLA
import pytest
import modin.numpy as np
import modin.numpy.linalg as LA
import modin.pandas as pd
from .utils import assert_scalar_or_array_equal
def test_norm_fro_1d():
    x1 = numpy.random.randint(-10, 10, size=100)
    numpy_result = NLA.norm(x1)
    x1 = np.array(x1)
    modin_result = LA.norm(x1)
    numpy.testing.assert_allclose(modin_result, numpy_result, rtol=1e-12)