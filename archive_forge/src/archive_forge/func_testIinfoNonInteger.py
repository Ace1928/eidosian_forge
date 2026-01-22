from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def testIinfoNonInteger(self):
    with self.assertRaises(ValueError):
        ml_dtypes.iinfo(np.float32)
    with self.assertRaises(ValueError):
        ml_dtypes.iinfo(np.complex128)
    with self.assertRaises(ValueError):
        ml_dtypes.iinfo(bool)