import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def generate_tests(self):
    tests = []
    for case in self.cases:
        for solver in _SOLVERS:
            if solver in case.skip:
                continue
            if solver in case.nonconvergence:
                tests += [SingleTest(case.A, case.b, solver, case.name, convergence=False)]
            else:
                tests += [SingleTest(case.A, case.b, solver, case.name)]
    return tests