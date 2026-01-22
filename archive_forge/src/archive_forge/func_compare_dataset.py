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
def compare_dataset(cls, el1, el2, msg='Dataset'):
    cls.compare_dimensioned(el1, el2)
    tabular = not (el1.interface.gridded and el2.interface.gridded)
    dimension_data = [(d, el1.dimension_values(d, expanded=tabular), el2.dimension_values(d, expanded=tabular)) for d in el1.kdims]
    dimension_data += [(d, el1.dimension_values(d, flat=tabular), el2.dimension_values(d, flat=tabular)) for d in el1.vdims]
    if el1.shape[0] != el2.shape[0]:
        raise AssertionError('%s not of matching length, %d vs. %d.' % (msg, el1.shape[0], el2.shape[0]))
    for dim, d1, d2 in dimension_data:
        with contextlib.suppress(Exception):
            np.testing.assert_equal(d1, d2)
            continue
        if d1.dtype != d2.dtype:
            failure_msg = f'{msg} {dim.pprint_label} columns have different type. First has type {d1}, and second has type {d2}.'
            raise cls.failureException(failure_msg)
        if d1.dtype.kind in 'SUOV':
            if list(d1) == list(d2):
                failure_msg = f'{msg} along dimension {dim.pprint_label} not equal.'
                raise cls.failureException(failure_msg)
        else:
            cls.compare_arrays(d1, d2, msg)