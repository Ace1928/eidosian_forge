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
def test_dask_backend_keywords(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            with parallel_config(backend='dask', workers=a['address']):
                seq = Parallel()((delayed(_worker_address)(i) for i in range(10)))
                assert seq == [a['address']] * 10
            with parallel_config(backend='dask', workers=b['address']):
                seq = Parallel()((delayed(_worker_address)(i) for i in range(10)))
                assert seq == [b['address']] * 10