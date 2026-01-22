import os
import sys
import warnings
import numpy as np
import pytest
from ..casting import sctypes
from ..testing import (
def test_warn_ignore():
    n_warns = len(warnings.filters)
    with suppress_warnings():
        warnings.warn('Here is a warning, you will not see it')
        warnings.warn('Nor this one', DeprecationWarning)
    with suppress_warnings() as w:
        warnings.warn('Here is a warning, you will not see it')
        warnings.warn('Nor this one', DeprecationWarning)
    assert n_warns == len(warnings.filters)

    def f():
        with suppress_warnings():
            raise ValueError('An error')
    with pytest.raises(ValueError):
        f()