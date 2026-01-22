from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_start_state_with_tasks_no_deps():
    dsk = {'a': [1, (inc, 2)], 'b': [1, 2, 3, 4], 'c': (inc, 3)}
    state = start_state_from_dask(dsk)
    assert list(state['cache'].keys()) == ['b']
    assert 'a' in state['ready'] and 'c' in state['ready']
    deps = {k: set() for k in 'abc'}
    assert state['dependencies'] == deps
    assert state['dependents'] == deps