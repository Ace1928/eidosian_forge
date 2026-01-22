import itertools
import random
import numpy as np
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
def test_router_bad_args():
    circuit = cirq.Circuit()
    device_graph = ccr.get_linear_device_graph(5)
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph)
    algo_name = 'greedy'
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name, router=ccr.ROUTERS[algo_name])
    circuit = cirq.Circuit(cirq.CCZ(*cirq.LineQubit.range(3)))
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name)
    circuit = cirq.Circuit((cirq.CZ(cirq.LineQubit(i), cirq.LineQubit(i + 1)) for i in range(5)))
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name)