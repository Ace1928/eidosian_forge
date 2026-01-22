from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
@pytest.mark.parametrize('bundle', [directly_connect_edges, hammer_bundle])
@pytest.mark.parametrize('layout', [random_layout, circular_layout, forceatlas2_layout])
def test_same_path_endpoints(layout, bundle):
    edges = pd.DataFrame({'id': [0], 'source': [0], 'target': [1]}).set_index('id')
    nodes = pd.DataFrame({'id': np.unique(edges.values)}).set_index('id')
    node_positions = layout(nodes, edges)
    bundled = bundle(node_positions, edges)
    source, target = edges.iloc[0]
    expected_source = node_positions.loc[source]
    expected_target = node_positions.loc[target]
    actual_source = bundled.loc[0]
    actual_target = bundled.loc[len(bundled) - 2]
    assert np.allclose(expected_source, actual_source)
    assert np.allclose(expected_target, actual_target)