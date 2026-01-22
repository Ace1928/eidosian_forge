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
@pytest.mark.parametrize('layers', [False, True])
def test_bind(layers):
    dsk1 = {('a-1', h1): 1, ('a-1', h2): 2}
    dsk2 = {'b-1': (add, ('a-1', h1), ('a-1', h2))}
    dsk3 = {'c-1': 'b-1'}
    cnt = NodeCounter()
    dsk4 = {('d-1', h1): (cnt.f, 1), ('d-1', h2): (cnt.f, 2)}
    dsk4b = {'e': (cnt.f, 3)}
    if layers:
        dsk1 = HighLevelGraph({'a-1': dsk1}, {'a-1': set()})
        dsk2 = HighLevelGraph({'a-1': dsk1, 'b-1': dsk2}, {'a-1': set(), 'b-1': {'a-1'}})
        dsk3 = HighLevelGraph({'a-1': dsk1, 'b-1': dsk2, 'c-1': dsk3}, {'a-1': set(), 'b-1': {'a-1'}, 'c-1': {'b-1'}})
        dsk4 = HighLevelGraph({'d-1': dsk4, 'e': dsk4b}, {'d-1': set(), 'e': set()})
    else:
        dsk2.update(dsk1)
        dsk3.update(dsk2)
        dsk4.update(dsk4b)
    t2 = Tuple(dsk2, ['b-1'])
    t3 = Tuple(dsk3, ['c-1'])
    t4 = Tuple(dsk4, [('d-1', h1), ('d-1', h2), 'e'])
    bound1 = bind(t3, t4, seed=1, assume_layers=layers)
    cloned_a_name = clone_key('a-1', seed=1)
    assert bound1.__dask_graph__()[cloned_a_name, h1][0] is chunks.bind
    assert bound1.__dask_graph__()[cloned_a_name, h2][0] is chunks.bind
    assert bound1.compute() == (3,)
    assert cnt.n == 3
    bound2 = bind(t3, t4, omit=t2, seed=1, assume_layers=layers)
    cloned_c_name = clone_key('c-1', seed=1)
    assert bound2.__dask_graph__()[cloned_c_name][0] is chunks.bind
    assert bound2.compute() == (3,)
    assert cnt.n == 6
    bound3 = bind(t4, t3, seed=1, assume_layers=layers)
    cloned_d_name = clone_key('d-1', seed=1)
    cloned_e_name = clone_key('e', seed=1)
    assert bound3.__dask_graph__()[cloned_d_name, h1][0] is chunks.bind
    assert bound3.__dask_graph__()[cloned_d_name, h2][0] is chunks.bind
    assert bound3.__dask_graph__()[cloned_e_name][0] is chunks.bind
    assert bound3.compute() == (1, 2, 3)
    assert cnt.n == 9