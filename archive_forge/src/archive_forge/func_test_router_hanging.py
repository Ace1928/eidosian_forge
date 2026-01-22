from multiprocessing import Process
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.routing.greedy import route_circuit_greedily
def test_router_hanging():
    """Run a separate process and check if greedy router hits timeout (5s)."""
    circuit, device_graph = create_circuit_and_device()
    process = Process(target=create_hanging_routing_instance, args=[circuit, device_graph])
    process.start()
    process.join(timeout=5)
    try:
        assert not process.is_alive(), 'Greedy router timeout'
    finally:
        process.terminate()