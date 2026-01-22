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
def test_cbc_lp_logging(self, capfd: pytest.LogCaptureFixture) -> None:
    """Validate that logLevel parameter is passed to solver"""
    fflush()
    capfd.readouterr()
    StandardTestLPs.test_lp_0(solver='CBC', duals=False, logLevel=0)
    fflush()
    quiet_output = capfd.readouterr()
    StandardTestLPs.test_lp_0(solver='CBC', duals=False, logLevel=5)
    fflush()
    verbose_output = capfd.readouterr()
    assert len(verbose_output.out) > len(quiet_output.out)
    StandardTestLPs.test_mi_lp_0(solver='CBC', logLevel=0)
    fflush()
    quiet_output = capfd.readouterr()
    StandardTestLPs.test_mi_lp_0(solver='CBC', logLevel=5)
    fflush()
    verbose_output = capfd.readouterr()
    assert len(verbose_output.out) > len(quiet_output.out)