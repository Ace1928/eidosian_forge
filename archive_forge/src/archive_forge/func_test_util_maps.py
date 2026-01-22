import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import tempfile
from pyomo.contrib.pynumero.interfaces.utils import (
def test_util_maps(self):
    anlp = AslNLP(self.filename)
    full_to_compressed_mask = build_compression_mask_for_finite_values(anlp.primals_lb())
    self.assertTrue(np.array_equal(full_to_compressed_mask, build_bounds_mask(anlp.primals_lb())))
    expected_compressed_primals_lb = np.asarray([-1, 2, -3, -5, -7, -9], dtype=np.float64)
    C = build_compression_matrix(full_to_compressed_mask)
    compressed_primals_lb = C * anlp.primals_lb()
    self.assertTrue(np.array_equal(expected_compressed_primals_lb, compressed_primals_lb))
    compressed_primals_lb = full_to_compressed(anlp.primals_lb(), full_to_compressed_mask)
    self.assertTrue(np.array_equal(expected_compressed_primals_lb, compressed_primals_lb))
    compressed_primals_lb = np.zeros(len(expected_compressed_primals_lb))
    ret = full_to_compressed(anlp.primals_lb(), full_to_compressed_mask, out=compressed_primals_lb)
    self.assertTrue(ret is compressed_primals_lb)
    self.assertTrue(np.array_equal(expected_compressed_primals_lb, compressed_primals_lb))
    expected_full_primals_lb = np.asarray([-1, 2, -3, -np.inf, -5, -np.inf, -7, -np.inf, -9], dtype=np.float64)
    full_primals_lb = compressed_to_full(compressed_primals_lb, full_to_compressed_mask, default=-np.inf)
    self.assertTrue(np.array_equal(expected_full_primals_lb, full_primals_lb))
    full_primals_lb.fill(0.0)
    ret = compressed_to_full(compressed_primals_lb, full_to_compressed_mask, out=full_primals_lb, default=-np.inf)
    self.assertTrue(ret is full_primals_lb)
    self.assertTrue(np.array_equal(expected_full_primals_lb, full_primals_lb))
    expected_full_primals_lb = np.asarray([-1, 2, -3, np.nan, -5, np.nan, -7, np.nan, -9], dtype=np.float64)
    full_primals_lb = compressed_to_full(compressed_primals_lb, full_to_compressed_mask)
    print(expected_full_primals_lb)
    print(full_primals_lb)
    np.testing.assert_array_equal(expected_full_primals_lb, full_primals_lb)
    expected_full_primals_lb = np.asarray([-1, 2, -3, 0.0, -5, 0.0, -7, 0.0, -9], dtype=np.float64)
    full_primals_lb.fill(0.0)
    ret = compressed_to_full(compressed_primals_lb, full_to_compressed_mask, out=full_primals_lb)
    self.assertTrue(ret is full_primals_lb)
    self.assertTrue(np.array_equal(expected_full_primals_lb, full_primals_lb))