import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_non_dcp_expression_is_not_dpp(self) -> None:
    x = cp.Parameter()
    expr = cp.exp(cp.log(x))
    self.assertFalse(expr.is_dpp())