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
@pytest.mark.skipif('not da')
@pytest.mark.parametrize('literal', [1, (1,), [1], {1: 1}, {1}])
def test_blockwise_clone_with_literals(literal):
    """https://github.com/dask/dask/issues/8978

    clone() on the result of a dask.array.blockwise operation with a (iterable) literal
    argument
    """
    arr = da.ones(10, chunks=1)

    def noop(arr, lit):
        return arr
    blk = da.blockwise(noop, 'x', arr, 'x', literal, None)
    cln = clone(blk)
    assert_no_common_keys(blk, cln, layers=True)
    da.utils.assert_eq(blk, cln)