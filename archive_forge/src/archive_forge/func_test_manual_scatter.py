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
def test_manual_scatter(loop):
    x = CountSerialized(1)
    y = CountSerialized(2)
    z = CountSerialized(3)
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            with parallel_config(backend='dask', scatter=[x, y]):
                f = delayed(add5)
                tasks = [f(x, y, z, d=4, e=5), f(x, z, y, d=5, e=4), f(y, x, z, d=x, e=5), f(z, z, x, d=z, e=y)]
                expected = [func(*args, **kwargs) for func, args, kwargs in tasks]
                results = Parallel()(tasks)
            with pytest.raises(TypeError):
                with parallel_config(backend='dask', loop=loop, scatter=1):
                    pass
    assert results == expected
    assert x.count == 1
    assert y.count == 1
    assert z.count in (4, 6)