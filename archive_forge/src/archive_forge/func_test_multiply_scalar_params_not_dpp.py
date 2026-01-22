import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_multiply_scalar_params_not_dpp(self) -> None:
    x = cp.Parameter()
    product = x * x
    self.assertFalse(product.is_dpp())
    self.assertTrue(product.is_dcp())