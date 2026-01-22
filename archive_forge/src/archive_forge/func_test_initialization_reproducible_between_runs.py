import random
import numpy as np
import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def test_initialization_reproducible_between_runs():
    seed = 45
    logical_graph = nx.erdos_renyi_graph(6, 0.5, seed=seed)
    logical_graph = nx.relabel_nodes(logical_graph, cirq.LineQubit)
    device_graph = ccr.get_grid_device_graph(2, 3)
    initial_mapping = ccr.initialization.get_initial_mapping(logical_graph, device_graph, seed)
    expected_mapping = {cirq.GridQubit(0, 0): cirq.LineQubit(5), cirq.GridQubit(0, 1): cirq.LineQubit(0), cirq.GridQubit(0, 2): cirq.LineQubit(2), cirq.GridQubit(1, 0): cirq.LineQubit(3), cirq.GridQubit(1, 1): cirq.LineQubit(4), cirq.GridQubit(1, 2): cirq.LineQubit(1)}
    assert initial_mapping == expected_mapping