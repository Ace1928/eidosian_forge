import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def test_tv_inpainting(self) -> None:
    self.skipTest('This benchmark takes too long.')
    if os.name == 'nt':
        self.skipTest('Skipping test due to overflow bug in SciPy < 1.2.0.')
    Uorig = np.random.randn(512, 512, 3)
    rows, cols, colors = Uorig.shape
    known = np.zeros((rows, cols, colors))
    for i in range(rows):
        for j in range(cols):
            if np.random.random() > 0.7:
                for k in range(colors):
                    known[i, j, k] = 1

    def tv_inpainting():
        Ucorr = known * Uorig
        variables = []
        constraints = []
        for i in range(colors):
            U = cp.Variable(shape=(rows, cols))
            variables.append(U)
            constraints.append(cp.multiply(known[:, :, i], U) == cp.multiply(known[:, :, i], Ucorr[:, :, i]))
        problem = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
        problem.get_problem_data(cp.SCS)
    benchmark(tv_inpainting, iters=1)