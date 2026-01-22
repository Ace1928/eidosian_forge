from __future__ import annotations
import traceback
from contextlib import contextmanager
import pytest
import dask
from dask.utils import shorten_traceback
def test_distributed_shorten_traceback():
    distributed = pytest.importorskip('distributed')
    with distributed.Client(processes=False, dashboard_address=':0'):
        d = dask.delayed(f3)()
        dp1, = dask.persist(d)
        dp2 = d.persist()
        TEST_NAME = 'test_distributed_shorten_traceback'
        expect = [TEST_NAME, 'compute', 'f3', 'f2', 'f1']
        with assert_tb_levels(expect):
            dask.compute(d())
        expect = [TEST_NAME, 'compute', 'compute', 'f3', 'f2', 'f1']
        with assert_tb_levels(expect):
            d.compute()
        expect = [TEST_NAME, 'compute', 'f3', 'f2', 'f1']
        with assert_tb_levels(expect):
            dask.compute(dp1)
        expect = [TEST_NAME, 'compute', 'compute', 'f3', 'f2', 'f1']
        with assert_tb_levels(expect):
            dp2.compute()