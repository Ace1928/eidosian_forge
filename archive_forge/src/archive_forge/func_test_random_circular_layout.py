from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
def test_random_circular_layout(nodes_without_positions, edges):
    expected_x = [0.066430214119, 0.407310906119, 0.99999987089, 0.338539010529, 0.802076237875]
    expected_y = [0.749032609855, 0.008666374166, 0.500359319055, 0.973212794501, 0.898434369139]
    df = circular_layout(nodes_without_positions, edges, uniform=False, seed=1)
    assert np.allclose(df['x'], expected_x)
    assert np.allclose(df['y'], expected_y)