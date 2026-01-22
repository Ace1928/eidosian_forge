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
def test_nested_backend_context_manager_implicit_n_jobs(loop):

    def _backend_type(p):
        return p._backend.__class__.__name__

    def get_nested_implicit_n_jobs():
        with Parallel() as p:
            return (_backend_type(p), p.n_jobs)
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            with parallel_config(backend='dask'):
                with Parallel() as p:
                    assert _backend_type(p) == 'DaskDistributedBackend'
                    assert p.n_jobs == -1
                    all_nested_n_jobs = p((delayed(get_nested_implicit_n_jobs)() for _ in range(2)))
                for backend_type, nested_n_jobs in all_nested_n_jobs:
                    assert backend_type == 'DaskDistributedBackend'
                    assert nested_n_jobs == -1