from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_start_state_with_independent_but_runnable_tasks():
    assert start_state_from_dask({'x': (inc, 1)})['ready'] == ['x']