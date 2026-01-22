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
def test_to_graphviz_custom():
    g = to_graphviz(dsk, data_attributes={'a': {'shape': 'square'}}, function_attributes={'c': {'label': 'neg_c', 'shape': 'ellipse'}})
    labels = set(filter(None, map(get_label, g.body)))
    assert labels == {'neg_c', 'd', 'e', 'f', '""'}
    shapes = list(filter(None, map(get_shape, g.body)))
    assert set(shapes) == {'box', 'circle', 'square', 'ellipse'}