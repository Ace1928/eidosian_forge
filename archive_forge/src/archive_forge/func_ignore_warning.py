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
@contextlib.contextmanager
def ignore_warning(**kw):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', **kw)
        yield