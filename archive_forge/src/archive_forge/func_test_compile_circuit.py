from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_compile_circuit():
    """Tests that we are able to compile a model circuit."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)])])
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(model_circuit, device_graph=ccr.gridqubits_to_graph_device(FakeDevice().qubits), compiler=compiler_mock, routing_attempts=1)
    assert len(compilation_result.mapping) == 3
    assert cirq.contrib.routing.ops_are_consistent_with_device_graph(compilation_result.circuit.all_operations(), cirq.contrib.routing.gridqubits_to_graph_device(FakeDevice().qubits))
    compiler_mock.assert_called_with(compilation_result.circuit)