import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def qp():
    x = cp.Variable(n)
    cp.Problem(cp.Minimize(1 / 2 * cp.quad_form(x, P) + cp.matmul(q.T, x)), [cp.matmul(G, x) <= h, cp.matmul(A, x) == b]).get_problem_data(cp.OSQP)