import ctypes
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
def test_do_backsolve(self):
    ma57 = MA57Interface()
    n = 5
    ne = 7
    irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
    jcn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
    irn = irn - 1
    jcn = jcn - 1
    ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0], dtype=np.double)
    rhs = np.array([8.0, 45.0, 31.0, 15.0, 17.0], dtype=np.double)
    status = ma57.do_symbolic_factorization(n, irn, jcn)
    status = ma57.do_numeric_factorization(n, ent)
    sol = ma57.do_backsolve(rhs)
    expected_sol = [1, 2, 3, 4, 5]
    old_rhs = np.array([8.0, 45.0, 31.0, 15.0, 17.0])
    for i in range(n):
        self.assertAlmostEqual(sol[i], expected_sol[i])
        self.assertEqual(old_rhs[i], rhs[i])