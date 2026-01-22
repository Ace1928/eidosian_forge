from __future__ import print_function, division, absolute_import
import os
import warnings
import pytest
from random import random
from uuid import uuid4
from time import sleep
from .. import Parallel, delayed, parallel_config
from ..parallel import ThreadingBackend, AutoBatchingMixin
from .._dask import DaskDistributedBackend
from distributed import Client, LocalCluster, get_client  # noqa: E402
from distributed.metrics import time  # noqa: E402
from distributed.utils_test import cluster, inc  # noqa: E402
@pytest.mark.parametrize('retry_no', list(range(2)))
def test_nested_scatter(loop, retry_no):
    np = pytest.importorskip('numpy')
    NUM_INNER_TASKS = 10
    NUM_OUTER_TASKS = 10

    def my_sum(x, i, j):
        return np.sum(x)

    def outer_function_joblib(array, i):
        client = get_client()
        with parallel_config(backend='dask'):
            results = Parallel()((delayed(my_sum)(array[j:], i, j) for j in range(NUM_INNER_TASKS)))
        return sum(results)
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as _:
            with parallel_config(backend='dask'):
                my_array = np.ones(10000)
                _ = Parallel()((delayed(outer_function_joblib)(my_array[i:], i) for i in range(NUM_OUTER_TASKS)))