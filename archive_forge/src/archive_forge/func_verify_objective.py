import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def verify_objective(self, places) -> None:
    actual = self.prob.value
    expect = self.expect_val
    if expect is not None:
        self.tester.assertAlmostEqual(actual, expect, places)