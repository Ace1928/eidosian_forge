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
def testRoundTripToInt(self, float_type):
    for v in INT_VALUES[float_type]:
        self.assertEqual(v, int(float_type(v)))
        self.assertEqual(-v, int(float_type(-v)))