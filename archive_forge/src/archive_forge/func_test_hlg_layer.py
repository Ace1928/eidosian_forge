from __future__ import annotations
import warnings
import pytest
from dask import utils_test
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import _check_warning
def test_hlg_layer():
    a = {'x': 1}
    b = {'y': (utils_test.inc, 'x')}
    layers = {'a-layer': a, 'bee-layer': b}
    dependencies = {'a-layer': set(), 'bee-layer': {'a-layer'}}
    hg = HighLevelGraph(layers, dependencies)
    assert utils_test.hlg_layer(hg, 'a') is hg.layers['a-layer']
    assert utils_test.hlg_layer(hg, 'b') is hg.layers['bee-layer']
    with pytest.raises(KeyError, match='No layer starts with'):
        utils_test.hlg_layer(hg, 'foo')