from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
@pytest.mark.parametrize('bundle', [directly_connect_edges, hammer_bundle])
def test_renamed_columns(nodes, weighted_edges, bundle):
    nodes = nodes.rename(columns={'x': 'xx', 'y': 'yy'})
    edges = weighted_edges.rename(columns={'source': 'src', 'target': 'dst', 'weight': 'w'})
    df = bundle(nodes, edges, x='xx', y='yy', source='src', target='dst', weight='w')
    assert 'xx' in df and 'x' not in df
    assert 'yy' in df and 'y' not in df
    assert 'w' in df and 'weight' not in df