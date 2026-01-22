import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def univar_obj(w0):
    return np.linalg.norm(D[:, 0] * w0 + D[:, 1] * (1 - w0), ord=p)