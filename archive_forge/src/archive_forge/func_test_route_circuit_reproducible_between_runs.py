import itertools
import random
import numpy as np
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
@pytest.mark.parametrize('algo', ccr.ROUTERS.keys())
def test_route_circuit_reproducible_between_runs(algo):
    seed = 23
    circuit = cirq.testing.random_circuit(6, 5, 0.5, random_state=seed)
    device_graph = ccr.get_grid_device_graph(2, 3)
    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo, random_state=seed)
    swap_network_str = str(swap_network).lstrip('\n').rstrip()
    expected_swap_network_str = '\n               ┌──┐       ┌────┐       ┌──────┐\n(0, 0): ───4────Z─────4────@───────4──────────────4───\n                           │\n(0, 1): ───2────@─────2────┼1↦0────5────@─────────5───\n                │          ││           │\n(0, 2): ───5────┼─────5────┼0↦1────2────┼iSwap────2───\n                │          │            ││\n(1, 0): ───3────┼T────3────@───────3────┼┼────────3───\n                │                       ││\n(1, 1): ───1────@─────1────────────1────X┼────────1───\n                                         │\n(1, 2): ───0────X─────0────────────0─────iSwap────0───\n               └──┘       └────┘       └──────┘\n    '.lstrip('\n').rstrip()
    assert swap_network_str == expected_swap_network_str