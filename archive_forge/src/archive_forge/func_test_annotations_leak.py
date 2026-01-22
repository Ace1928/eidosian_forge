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
def test_annotations_leak():
    """Annotations shouldn't leak between threads.
    See https://github.com/dask/dask/issues/10340."""
    b1 = threading.Barrier(2)
    b2 = threading.Barrier(2)

    def f(n):
        with dask.annotate(foo=n):
            b1.wait()
            out = dask.get_annotations()
            b2.wait()
            return out
    with ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(f, 1)
        f2 = ex.submit(f, 2)
        result = [f1.result(), f2.result()]
    assert result == [{'foo': 1}, {'foo': 2}]