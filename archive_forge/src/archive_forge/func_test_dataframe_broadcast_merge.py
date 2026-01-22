from __future__ import annotations
import pytest
import asyncio
import os
import sys
from functools import partial
from operator import add
from distributed import Client, SchedulerPlugin, WorkerPlugin
from distributed.utils_test import cleanup  # noqa F401
from distributed.utils_test import client as c  # noqa F401
from distributed.utils_test import (  # noqa F401
import dask
import dask.bag as db
from dask import compute, delayed, persist
from dask.base import compute_as_if_collection, get_scheduler
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.distributed import futures_of, wait
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.utils import get_named_args, get_scheduler_lock, tmpdir, tmpfile
from dask.utils_test import inc
@pytest.mark.parametrize('on', ['a', ['a']])
@pytest.mark.parametrize('broadcast', [True, False])
def test_dataframe_broadcast_merge(c, on, broadcast):
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    pdfl = pd.DataFrame({'a': [1, 2] * 2, 'b_left': range(4)})
    pdfr = pd.DataFrame({'a': [2, 1], 'b_right': range(2)})
    dfl = dd.from_pandas(pdfl, npartitions=4)
    dfr = dd.from_pandas(pdfr, npartitions=2)
    ddfm = dd.merge(dfl, dfr, on=on, broadcast=broadcast, shuffle_method='tasks')
    dfm = ddfm.compute()
    dd.utils.assert_eq(dfm.sort_values('a'), pd.merge(pdfl, pdfr, on=on).sort_values('a'), check_index=False)