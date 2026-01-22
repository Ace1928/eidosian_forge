import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.product(scalar_type=INT4_TYPES, op=[operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod])
def testBinop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
        for w in VALUES[scalar_type]:
            if w == 0 and op in [operator.floordiv, operator.mod]:
                with self.assertRaises(ZeroDivisionError):
                    op(scalar_type(v), scalar_type(w))
            else:
                out = op(scalar_type(v), scalar_type(w))
                self.assertIsInstance(out, scalar_type)
                self.assertEqual(scalar_type(op(v, w)), out, msg=(v, w))