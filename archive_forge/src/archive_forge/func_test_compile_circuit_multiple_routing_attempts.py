from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_compile_circuit_multiple_routing_attempts():
    """Tests that we make multiple attempts at routing and keep the best one."""
    qubits = cirq.LineQubit.range(3)
    initial_mapping = dict(zip(qubits, qubits))
    more_operations = cirq.Circuit([cirq.X.on_each(qubits), cirq.Y.on_each(qubits)])
    more_qubits = cirq.Circuit([cirq.X.on_each(cirq.LineQubit.range(4))])
    well_routed = cirq.Circuit([cirq.X.on_each(qubits)])
    router_mock = MagicMock(side_effect=[ccr.SwapNetwork(more_operations, initial_mapping), ccr.SwapNetwork(well_routed, initial_mapping), ccr.SwapNetwork(more_qubits, initial_mapping)])
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    model_circuit = cirq.Circuit([cirq.X.on_each(qubits)])
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(model_circuit, device_graph=ccr.gridqubits_to_graph_device(FakeDevice().qubits), compiler=compiler_mock, router=router_mock, routing_attempts=3)
    assert compilation_result.mapping == initial_mapping
    assert router_mock.call_count == 3
    compiler_mock.assert_called_with(well_routed)