import collections
import contextlib
import copy
import itertools
import math
import pickle
import sys
from typing import Type
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def testHashNan(self, float_type):
    for name, nan in [('PositiveNan', float_type(float('nan'))), ('NegativeNan', float_type(float('-nan')))]:
        with self.subTest(name):
            nan_hash = hash(nan)
            nan_object_hash = object.__hash__(nan)
            self.assertIn(nan_hash, (sys.hash_info.nan, nan_object_hash), str(nan))