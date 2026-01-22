from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_get_sync_num_workers(self):
    self.get({'x': (inc, 'y'), 'y': 1}, 'x', num_workers=2)