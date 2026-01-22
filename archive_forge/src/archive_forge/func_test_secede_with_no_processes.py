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
def test_secede_with_no_processes(loop):
    with Client(loop=loop, processes=False, set_as_default=True):
        with parallel_config(backend='dask'):
            Parallel(n_jobs=4)((delayed(id)(i) for i in range(2)))