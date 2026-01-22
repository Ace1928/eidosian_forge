import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.product(scalar_type=INT4_TYPES, op=[operator.le, operator.lt, operator.eq, operator.ne, operator.ge, operator.gt])
def testComparison(self, scalar_type, op):
    for v in VALUES[scalar_type]:
        for w in VALUES[scalar_type]:
            self.assertEqual(op(v, w), op(scalar_type(v), scalar_type(w)))