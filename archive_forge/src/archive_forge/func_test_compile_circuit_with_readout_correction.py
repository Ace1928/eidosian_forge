from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_compile_circuit_with_readout_correction():
    """Tests that we are able to compile a model circuit with readout error
    correction."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    router_mock = MagicMock(side_effect=lambda circuit, network: ccr.SwapNetwork(circuit, {}))
    a, b, c = cirq.LineQubit.range(3)
    ap, bp, cp = cirq.LineQubit.range(3, 6)
    model_circuit = cirq.Circuit([cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)])])
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(model_circuit, device_graph=ccr.gridqubits_to_graph_device(FakeDevice().qubits), compiler=compiler_mock, router=router_mock, routing_attempts=1, add_readout_error_correction=True)
    assert compilation_result.circuit == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]), cirq.Moment([cirq.X(a), cirq.X(b), cirq.X(c)]), cirq.Moment([cirq.CNOT(a, ap), cirq.CNOT(b, bp), cirq.CNOT(c, cp)]), cirq.Moment([cirq.X(a), cirq.X(b), cirq.X(c)])])