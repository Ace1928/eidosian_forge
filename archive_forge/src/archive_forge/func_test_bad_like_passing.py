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
def test_bad_like_passing(self):

    def func(*, like=None):
        pass
    func_with_like = array_function_dispatch()(func)
    with pytest.raises(TypeError):
        func_with_like()
    with pytest.raises(TypeError):
        func_with_like(like=234)