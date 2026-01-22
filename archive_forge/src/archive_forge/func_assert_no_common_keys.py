from __future__ import annotations
import random
import time
from operator import add
import pytest
import dask
import dask.bag as db
from dask import delayed
from dask.base import clone_key
from dask.blockwise import Blockwise
from dask.graph_manipulation import bind, checkpoint, chunks, clone, wait_on
from dask.highlevelgraph import HighLevelGraph
from dask.tests.test_base import Tuple
from dask.utils_test import import_or_none
def assert_no_common_keys(a, b, omit=None, *, layers: bool) -> None:
    dsk1 = a.__dask_graph__()
    dsk2 = b.__dask_graph__()
    if omit is not None:
        dsko = omit.__dask_graph__()
        assert not dsk1.keys() - dsko.keys() & dsk2.keys()
        assert not dsko.keys() - dsk1.keys()
        assert not dsko.keys() - dsk2.keys()
        if layers:
            assert not dsk1.layers.keys() - dsko.layers.keys() & dsk2.layers.keys()
            assert not dsk1.dependencies.keys() - dsko.dependencies.keys() & dsk2.dependencies.keys()
            assert not dsko.layers.keys() - dsk1.layers.keys()
            assert not dsko.layers.keys() - dsk2.layers.keys()
            assert not dsko.dependencies.keys() - dsk1.dependencies.keys()
            assert not dsko.dependencies.keys() - dsk2.dependencies.keys()
    else:
        assert not dsk1.keys() & dsk2.keys()
        if layers:
            assert not dsk1.layers.keys() & dsk2.layers.keys()
            assert not dsk1.dependencies.keys() & dsk2.dependencies.keys()