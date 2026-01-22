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
def testRepr(self, float_type):
    for value in FLOAT_VALUES[float_type]:
        self.assertEqual('%.6g' % float(float_type(value)), repr(float_type(value)))