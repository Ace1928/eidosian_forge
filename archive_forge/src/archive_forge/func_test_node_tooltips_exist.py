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
def test_node_tooltips_exist():
    da = pytest.importorskip('dask.array')
    pytest.importorskip('graphviz')
    a = da.ones((1000, 1000), chunks=(100, 100))
    b = a + a.T
    c = b.sum(axis=1)
    hg = c.dask
    g = to_graphviz(hg)
    for layer in g.body:
        if 'label' in layer:
            assert 'tooltip' in layer
            start = layer.find('tooltip="') + len('tooltip="')
            end = layer.find('"', start)
            tooltip = layer[start:end]
            assert len(tooltip) > 0