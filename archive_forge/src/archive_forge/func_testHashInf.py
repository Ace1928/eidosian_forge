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
def testHashInf(self, float_type):
    if dtype_has_inf(float_type):
        self.assertEqual(sys.hash_info.inf, hash(float_type(float('inf'))), 'inf')
        self.assertEqual(-sys.hash_info.inf, hash(float_type(float('-inf'))), '-inf')