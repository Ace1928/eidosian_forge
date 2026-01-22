import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_matmul_params_not_dpp(self) -> None:
    X = cp.Parameter((4, 4))
    product = X @ X
    self.assertTrue(product.is_dcp())
    self.assertFalse(product.is_dpp())