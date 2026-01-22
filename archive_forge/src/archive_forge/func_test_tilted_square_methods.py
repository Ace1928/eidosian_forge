import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
def test_tilted_square_methods():
    topo = TiltedSquareLattice(5, 5)
    ax = MagicMock()
    topo.draw(ax=ax)
    ax.scatter.assert_called()
    qubits = topo.nodes_as_gridqubits()
    assert all((isinstance(q, cirq.GridQubit) for q in qubits))
    mapping = topo.nodes_to_gridqubits(offset=(3, 5))
    assert all((isinstance(q, cirq.GridQubit) and q >= cirq.GridQubit(0, 0) for q in mapping.values()))