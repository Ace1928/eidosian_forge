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
def test_cytoscape_graph_custom(tmp_path):
    g = cytoscape_graph(dsk, filename=os.fsdecode(tmp_path / 'mydask.html'), rankdir='LR', node_sep=20, edge_sep=30, spacing_factor=2, edge_style={'line-color': 'red'}, node_style={'background-color': 'green'})
    sty = g.cytoscape_style
    layout = g.cytoscape_layout
    node_sty = next((s for s in sty if s['selector'] == 'node'))['style']
    edge_sty = next((s for s in sty if s['selector'] == 'edge'))['style']
    assert edge_sty['line-color'] == 'red'
    assert node_sty['background-color'] == 'green'
    assert layout['rankDir'] == 'LR'
    assert layout['nodeSep'] == 20
    assert layout['edgeSep'] == 30
    assert layout['spacingFactor'] == 2