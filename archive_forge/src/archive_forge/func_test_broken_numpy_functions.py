import builtins
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.expressions.expression import (
from cvxpy.tests.base_test import BaseTest
def test_broken_numpy_functions(self) -> None:
    with pytest.raises(RuntimeError, match=__NUMPY_UFUNC_ERROR__):
        np.linalg.norm(self.x)