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
@pytest.mark.parametrize('func', [lambda x, y: 0, lambda like=None: 0, lambda *, like=None, a=3: 0])
def test_bad_like_sig(self, func):
    with pytest.raises(RuntimeError):
        array_function_dispatch()(func)