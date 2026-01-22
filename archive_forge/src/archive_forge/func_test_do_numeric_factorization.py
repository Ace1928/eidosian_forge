import ctypes
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
def test_do_numeric_factorization(self):
    ma57 = MA57Interface()
    n = 5
    ne = 7
    irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
    jcn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
    irn = irn - 1
    jcn = jcn - 1
    ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0], dtype=np.double)
    ma57.do_symbolic_factorization(n, irn, jcn)
    ma57.fact_factor = 1.5
    ma57.ifact_factor = 1.5
    status = ma57.do_numeric_factorization(n, ent)
    self.assertEqual(status, 0)
    self.assertEqual(ma57.get_info(14), 12)
    self.assertEqual(ma57.get_info(24), 2)
    self.assertEqual(ma57.get_info(22), 1)
    self.assertEqual(ma57.get_info(23), 0)
    ent2 = np.array([1.0, 5.0, 1.0, 6.0, 4.0, 3.0, 2.0], dtype=np.double)
    ma57.do_numeric_factorization(n, ent2)
    self.assertEqual(status, 0)
    bad_ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0], dtype=np.double)
    with self.assertRaisesRegex(AssertionError, 'Wrong number of entries'):
        ma57.do_numeric_factorization(n, bad_ent)
    with self.assertRaisesRegex(AssertionError, 'Dimension mismatch'):
        ma57.do_numeric_factorization(n + 1, ent)
    n = 5
    ne = 8
    irn = np.array([1, 1, 2, 2, 3, 3, 5, 5], dtype=np.intc)
    jcn = np.array([1, 2, 3, 5, 3, 4, 5, 1], dtype=np.intc)
    irn = irn - 1
    jcn = jcn - 1
    ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0, -1.3], dtype=np.double)
    status = ma57.do_symbolic_factorization(n, irn, jcn)
    self.assertEqual(status, 0)
    status = ma57.do_numeric_factorization(n, ent)
    self.assertEqual(status, 0)
    self.assertEqual(ma57.get_info(24), 2)
    self.assertEqual(ma57.get_info(23), 0)