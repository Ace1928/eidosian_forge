import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_multiply_nonlinear_nonneg_param_and_nonneg_variable_is_not_dpp(self) -> None:
    x = cp.Parameter(nonneg=True)
    y = cp.Variable(nonneg=True)
    product = cp.exp(x) * y
    self.assertFalse(product.is_dpp())
    self.assertTrue(product.is_dcp())