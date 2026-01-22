from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_subs_no_key_data_eq():
    a = MutateOnEq()
    subs(a, 'x', 1)
    assert a.hit_eq == 0
    subs((add, a, 'x'), 'x', 1)
    assert a.hit_eq == 0