import builtins
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.expressions.expression import (
from cvxpy.tests.base_test import BaseTest
def test_np_ufunc_errors(self) -> None:
    with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
        np.sqrt(self.x)
    with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
        np.log(self.x)