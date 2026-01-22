from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
def test_forceatlas2_unpositioned_nodes_with_unweighted_edges(nodes_without_positions, edges):
    df = forceatlas2_layout(nodes_without_positions, edges)
    assert len(nodes_without_positions) == len(df)
    assert not df.equals(nodes_without_positions)