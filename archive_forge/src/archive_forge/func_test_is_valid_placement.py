import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
def test_is_valid_placement():
    topo = TiltedSquareLattice(4, 2)
    syc23 = TiltedSquareLattice(8, 4).graph
    placements = get_placements(syc23, topo.graph)
    for placement in placements:
        assert is_valid_placement(syc23, topo.graph, placement)
    bad_placement = topo.nodes_to_gridqubits(offset=(100, 100))
    assert not is_valid_placement(syc23, topo.graph, bad_placement)