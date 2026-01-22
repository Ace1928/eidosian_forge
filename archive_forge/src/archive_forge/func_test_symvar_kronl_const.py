from typing import Tuple
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def test_symvar_kronl_const(self):
    self.symvar_kronl(param=False)