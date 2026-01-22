import contextlib
from functools import partial
from unittest import TestCase
from unittest.util import safe_repr
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..core import (
from ..core.options import Cycle, Options
from ..core.util import cast_array_to_int64, datetime_types, dt_to_int, is_float
from . import *  # noqa (All Elements need to support comparison)
@classmethod
def compare_dictionaries(cls, d1, d2, msg='Dictionaries'):
    keys = set(d1.keys())
    keys2 = set(d2.keys())
    symmetric_diff = keys ^ keys2
    if symmetric_diff:
        msg = f'Dictionaries have different sets of keys: {symmetric_diff!r}\n\n'
        msg += f'Dictionary 1: {d1}\n'
        msg += f'Dictionary 2: {d2}'
        raise cls.failureException(msg)
    for k in keys:
        cls.assertEqual(d1[k], d2[k])