import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
@unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
def test_scipy_mi_time_limit_reached(self) -> None:
    sth = sths.mi_lp_7()
    sth.solve(solver='SCIPY', scipy_options={'time_limit': 100.0})