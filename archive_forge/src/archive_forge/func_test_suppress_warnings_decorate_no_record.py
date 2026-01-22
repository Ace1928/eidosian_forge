import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_suppress_warnings_decorate_no_record():
    sup = suppress_warnings()
    sup.filter(UserWarning)

    @sup
    def warn(category):
        warnings.warn('Some warning', category)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        warn(UserWarning)
        warn(RuntimeWarning)
        assert_equal(len(w), 1)