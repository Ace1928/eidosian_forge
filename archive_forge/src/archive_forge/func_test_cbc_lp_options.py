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
@pytest.mark.parametrize('opts', [pytest.param(opts, id=next(iter(opts.keys()))) for opts in [{'dualTolerance': 1.0}, {'primalTolerance': 1.0}, {'maxNumIteration': 1}, {'scaling': 0}, {'optimizationDirection': 'max'}, {'presolve': 'off'}]])
def test_cbc_lp_options(self, opts: dict, capfd: pytest.LogCaptureFixture) -> None:
    """
        Validate that cylp is actually using each option.

        Tentative approach: run model with verbose output with or without the specified
        option; verbose output should be different each way.
        """
    fflush()
    capfd.readouterr()
    sth = sths.lp_4()
    sth.solve(solver='CBC', logLevel=2)
    fflush()
    base = capfd.readouterr()
    try:
        sth.solve(solver='CBC', logLevel=2, **opts)
    except Exception:
        pass
    else:
        fflush()
        with_opt = capfd.readouterr()
        assert base != with_opt