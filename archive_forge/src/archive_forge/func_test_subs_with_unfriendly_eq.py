from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_subs_with_unfriendly_eq():
    try:
        import numpy as np
    except ImportError:
        return
    else:
        task = (np.sum, np.array([1, 2]))
        assert (subs(task, (4, 5), 1) == task) is True

    class MyException(Exception):
        pass

    class F:

        def __eq__(self, other):
            raise MyException()
    task = F()
    assert subs(task, 1, 2) is task