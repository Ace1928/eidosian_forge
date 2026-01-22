import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def randn_psd(n):
    A = 1.0 / 10 * np.random.randn(n, n)
    return np.matmul(A, A.T)