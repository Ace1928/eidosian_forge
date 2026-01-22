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
@pytest.mark.skipif('not da or not dd')
def test_wait_on_collections():
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled():
        pytest.skip("hlg doesn't make sense")
    colls, cnt = collections_with_node_counters()

    @delayed
    def f(x):
        pass
    colls2 = wait_on(*colls)
    f(colls2[0]).compute()
    assert cnt.n == 16
    assert colls2[0].compute() == colls[0].compute()
    da.utils.assert_eq(colls2[1], colls[1])
    da.utils.assert_eq(colls2[2], colls[2])
    db.utils.assert_eq(colls2[3], colls[3])
    db.utils.assert_eq(colls2[4], colls[4])
    db.utils.assert_eq(colls2[5], colls[5])
    dd.utils.assert_eq(colls2[6], colls[6])
    dd.utils.assert_eq(colls2[7], colls[7])
    dd.utils.assert_eq(colls2[8], colls[8])
    dd.utils.assert_eq(colls2[9], colls[9])