import builtins
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.expressions.expression import (
from cvxpy.tests.base_test import BaseTest
def test_working_numpy_functions(self) -> None:
    hstack = np.hstack([self.x])
    self.assertEqual(hstack.shape, (1,))
    self.assertEqual(hstack.dtype, object)
    vstack = np.vstack([self.x])
    self.assertEqual(vstack.shape, (1, 1))
    self.assertEqual(vstack.dtype, object)