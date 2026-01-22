from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def testIinfoInt4(self):
    info = ml_dtypes.iinfo(ml_dtypes.int4)
    self.assertEqual(info.dtype, ml_dtypes.iinfo('int4').dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype('int4')).dtype)
    self.assertEqual(info.min, -8)
    self.assertEqual(info.max, 7)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.int4))
    self.assertEqual(info.bits, 4)
    self.assertEqual(info.kind, 'i')
    self.assertEqual(str(info), 'iinfo(min=-8, max=7, dtype=int4)')