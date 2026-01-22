from multiprocessing import Process
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.routing.greedy import route_circuit_greedily
def create_circuit_and_device():
    """Construct a small circuit and a device with line connectivity
    to test the greedy router. This instance hangs router in Cirq 8.2.
    """
    num_qubits = 6
    gate_domain = {cirq.ops.CNOT: 2}
    circuit = cirq.testing.random_circuit(num_qubits, 15, 0.5, gate_domain, random_state=37)
    device_graph = ccr.get_linear_device_graph(num_qubits)
    return (circuit, device_graph)