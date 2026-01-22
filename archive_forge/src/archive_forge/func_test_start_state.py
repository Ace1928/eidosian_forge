from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_start_state():
    dsk = {'x': 1, 'y': 2, 'z': (inc, 'x'), 'w': (add, 'z', 'y')}
    result = start_state_from_dask(dsk)
    expected = {'cache': {'x': 1, 'y': 2}, 'dependencies': {'w': {'y', 'z'}, 'x': set(), 'y': set(), 'z': {'x'}}, 'dependents': {'w': set(), 'x': {'z'}, 'y': {'w'}, 'z': {'w'}}, 'finished': set(), 'released': set(), 'running': set(), 'ready': ['z'], 'waiting': {'w': {'z'}}, 'waiting_data': {'x': {'z'}, 'y': {'w'}, 'z': {'w'}}}
    assert result == expected