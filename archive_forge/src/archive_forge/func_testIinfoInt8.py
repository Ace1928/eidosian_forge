from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def testIinfoInt8(self):
    info = ml_dtypes.iinfo(np.int8)
    self.assertEqual(info.min, -128)
    self.assertEqual(info.max, 127)