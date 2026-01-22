import osqp
import numpy as np
from scipy import sparse
import unittest
def test_issue14(self):
    m = osqp.OSQP()
    m.setup(self.P, self.q, self.A, self.l, self.u, linsys_solver='mkl pardiso')
    m.solve()