import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_scipy_trapz_support_shim():
    import types
    import functools

    def _copy_func(f):
        g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
        g = functools.update_wrapper(g, f)
        g.__kwdefaults__ = f.__kwdefaults__
        return g
    trapezoid = _copy_func(np.trapz)
    assert np.trapz([1, 2]) == trapezoid([1, 2])