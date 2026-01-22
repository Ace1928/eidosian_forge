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
def testDtypeFromString(self, float_type):
    assert np.dtype(float_type.__name__) == np.dtype(float_type)