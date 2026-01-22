from __future__ import annotations
import copy
import os
import re
import sys
from functools import partial
from operator import add, neg
import pytest
from dask.dot import _to_cytoscape_json, cytoscape_graph
from dask import delayed
from dask.utils import ensure_not_exists
@pytest.mark.parametrize('viz_func', [to_graphviz, _to_cytoscape_json])
def test_immutable_attributes(viz_func):

    def inc(x):
        return x + 1
    dsk = {'a': (inc, 1), 'b': (inc, 2), 'c': (add, 'a', 'b')}
    attrs_func = {'a': {}}
    attrs_data = {'b': {}}
    attrs_func_test = copy.deepcopy(attrs_func)
    attrs_data_test = copy.deepcopy(attrs_data)
    viz_func(dsk, function_attributes=attrs_func, data_attributes=attrs_data)
    assert attrs_func_test == attrs_func
    assert attrs_data_test == attrs_data