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
@pytest.mark.parametrize('cluster_strategy', ['adaptive', 'late_scaling'])
@pytest.mark.skipif(distributed.__version__ <= '2.1.1' and distributed.__version__ >= '1.28.0', reason='distributed bug - https://github.com/dask/distributed/pull/2841')
def test_wait_for_workers(cluster_strategy):
    cluster = LocalCluster(n_workers=0, processes=False, threads_per_worker=2)
    client = Client(cluster)
    if cluster_strategy == 'adaptive':
        cluster.adapt(minimum=0, maximum=2)
    elif cluster_strategy == 'late_scaling':
        cluster.scale(2)
    try:
        with parallel_config(backend='dask'):
            Parallel()((delayed(inc)(i) for i in range(10)))
    finally:
        client.close()
        cluster.close()