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
def demo_tuples(layers: bool) -> tuple[Tuple, Tuple, NodeCounter]:
    cnt = NodeCounter()
    dsk1 = HighLevelGraph({'a': {('a', h1): (cnt.f, 1), ('a', h2): (cnt.f, 2)}, 'b': {'b': (cnt.f, 3)}}, {'a': set(), 'b': set()})
    dsk2 = HighLevelGraph({'c': {'c': (cnt.f, 4)}, 'd': {'d': (cnt.f, 5)}}, {'c': set(), 'd': set()})
    if not layers:
        dsk1 = dsk1.to_dict()
        dsk2 = dsk2.to_dict()
    return (Tuple(dsk1, list(dsk1)), Tuple(dsk2, list(dsk2)), cnt)