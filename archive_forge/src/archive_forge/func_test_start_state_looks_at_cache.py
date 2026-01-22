from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_start_state_looks_at_cache():
    dsk = {'b': (inc, 'a')}
    cache = {'a': 1}
    result = start_state_from_dask(dsk, cache)
    assert result['dependencies']['b'] == {'a'}
    assert result['ready'] == ['b']