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
def testLessEqual(self, float_type):
    for v in FLOAT_VALUES[float_type]:
        for w in FLOAT_VALUES[float_type]:
            self.assertEqual(v <= w, float_type(v) <= float_type(w))