import osqp
from osqp._osqp import constant
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_update_max_iter(self):
    self.model.update_settings(max_iter=80)
    res = self.model.solve()
    self.assertEqual(res.info.status_val, constant('OSQP_MAX_ITER_REACHED'))