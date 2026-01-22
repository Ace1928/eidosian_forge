import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def verify_primal_values(self, places) -> None:
    for idx in range(len(self.variables)):
        actual = self.variables[idx].value
        expect = self.expect_prim_vars[idx]
        if expect is not None:
            self.tester.assertItemsAlmostEqual(actual, expect, places)