from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_calculate_quantum_volume_result():
    """Test that running the main loop returns the desired result"""
    results = cirq.contrib.quantum_volume.calculate_quantum_volume(num_qubits=3, depth=3, num_circuits=1, device_graph=ccr.gridqubits_to_graph_device(cirq.GridQubit.rect(3, 3)), samplers=[cirq.Simulator()], routing_attempts=2, random_state=1)
    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(3, 3, random_state=1)
    assert len(results) == 1
    assert results[0].model_circuit == model_circuit
    assert results[0].heavy_set == cirq.contrib.quantum_volume.compute_heavy_set(model_circuit)
    buffer = io.StringIO()
    cirq.to_json(results, buffer)