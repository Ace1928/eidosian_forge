from typing import List
import pytest
import cirq
from cirq_aqt.aqt_device_metadata import AQTDeviceMetadata
from cirq_aqt.aqt_target_gateset import AQTTargetGateset
def test_aqtdevice_metadata(metadata, qubits):
    assert metadata.qubit_set == frozenset(qubits)
    assert set(qubits) == set(metadata.nx_graph.nodes())
    edges = metadata.nx_graph.edges()
    assert len(edges) == 10
    assert all((q0 != q1 for q0, q1 in edges))
    assert AQTTargetGateset() == metadata.gateset
    assert len(metadata.gate_durations) == 6