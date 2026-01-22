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
def compare_dimension_lists(cls, dlist1, dlist2, msg='Dimension lists'):
    if len(dlist1) != len(dlist2):
        raise cls.failureException(f'{msg} mismatched')
    for d1, d2 in zip(dlist1, dlist2):
        cls.assertEqual(d1, d2)