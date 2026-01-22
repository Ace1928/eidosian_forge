import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_invalid_cases(self):
    for args in self.invalid:
        with pytest.raises(ValueError):
            tools.validate_vector_shape(*args)