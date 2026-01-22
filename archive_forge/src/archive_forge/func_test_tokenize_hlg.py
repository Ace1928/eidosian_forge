from __future__ import annotations
import os
import threading
import xml.etree.ElementTree
from collections.abc import Set
from concurrent.futures import ThreadPoolExecutor
import pytest
import dask
from dask.base import tokenize
from dask.blockwise import Blockwise, blockwise_token
from dask.highlevelgraph import HighLevelGraph, Layer, MaterializedLayer, to_graphviz
from dask.utils_test import inc
def test_tokenize_hlg():
    import dask.bag as db
    a = db.from_sequence(list(range(10)), npartitions=2).max()
    b = db.from_sequence(list(range(10)), npartitions=2).max()
    c = db.from_sequence(list(range(10)), npartitions=3).max()
    assert tokenize(a.dask) == tokenize(b.dask)
    assert tokenize(a.dask) != tokenize(c.dask)