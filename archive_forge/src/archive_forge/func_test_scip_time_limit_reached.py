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
def test_scip_time_limit_reached(self) -> None:
    sth = sths.mi_lp_7()
    with pytest.raises(cp.error.SolverError) as se:
        sth.solve(solver='SCIP', scip_params={'limits/time': 0.0})
        exc = "Solver 'SCIP' failed. Try another solver, or solve with verbose=True for more information."
        assert str(se.value) == exc