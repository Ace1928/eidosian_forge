import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.product(scalar_type=INT4_TYPES, op=[operator.neg, operator.pos])
def testUnop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
        out = op(scalar_type(v))
        self.assertIsInstance(out, scalar_type)
        self.assertEqual(scalar_type(op(v)), out, msg=v)