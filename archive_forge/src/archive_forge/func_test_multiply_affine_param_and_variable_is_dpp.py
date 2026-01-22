import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_multiply_affine_param_and_variable_is_dpp(self) -> None:
    x = cp.Parameter()
    y = cp.Variable()
    product = (x + x) * y
    self.assertTrue(product.is_dpp())
    self.assertTrue(product.is_dcp())