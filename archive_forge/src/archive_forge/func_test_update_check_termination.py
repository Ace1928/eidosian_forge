import osqp
from osqp._osqp import constant
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_update_check_termination(self):
    self.model.update_settings(check_termination=0)
    res = self.model.solve()
    self.assertEqual(res.info.iter, self.opts['max_iter'])