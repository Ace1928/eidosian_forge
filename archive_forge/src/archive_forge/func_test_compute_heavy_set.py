from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_compute_heavy_set():
    """Test that the heavy set can be computed from a given circuit."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([cirq.Moment([]), cirq.Moment([cirq.X(a), cirq.Y(b)]), cirq.Moment([]), cirq.Moment([cirq.CNOT(a, c)]), cirq.Moment([cirq.Z(a), cirq.H(b)])])
    assert cirq.contrib.quantum_volume.compute_heavy_set(model_circuit) == [5, 7]