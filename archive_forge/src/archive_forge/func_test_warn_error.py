import os
import sys
import warnings
import numpy as np
import pytest
from ..casting import sctypes
from ..testing import (
def test_warn_error():
    n_warns = len(warnings.filters)
    with error_warnings():
        with pytest.raises(UserWarning):
            warnings.warn('A test')
    with error_warnings() as w:
        with pytest.raises(UserWarning):
            warnings.warn('A test')
    assert n_warns == len(warnings.filters)

    def f():
        with error_warnings():
            raise ValueError('An error')
    with pytest.raises(ValueError):
        f()