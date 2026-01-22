import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_warn_wrong_warning(self):

    def f():
        warnings.warn('yo', DeprecationWarning)
    failed = False
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        try:
            assert_warns(UserWarning, f)
            failed = True
        except DeprecationWarning:
            pass
    if failed:
        raise AssertionError('wrong warning caught by assert_warn')