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
def test_glpk_mi_options(self) -> None:
    sth = sths.mi_lp_1()
    import cvxopt
    assert 'tm_lim' not in cvxopt.glpk.options
    sth.solve(solver='GLPK_MI', tm_lim=100, verbose=True)
    assert 'tm_lim' not in cvxopt.glpk.options
    sth.verify_objective(places=4)
    sth.verify_primal_values(places=4)