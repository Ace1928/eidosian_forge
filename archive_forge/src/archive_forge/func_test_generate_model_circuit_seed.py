from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_generate_model_circuit_seed():
    """Test that a model circuit is determined by its seed ."""
    model_circuit_1 = cirq.contrib.quantum_volume.generate_model_circuit(3, 3, random_state=np.random.RandomState(1))
    model_circuit_2 = cirq.contrib.quantum_volume.generate_model_circuit(3, 3, random_state=np.random.RandomState(1))
    model_circuit_3 = cirq.contrib.quantum_volume.generate_model_circuit(3, 3, random_state=np.random.RandomState(2))
    assert model_circuit_1 == model_circuit_2
    assert model_circuit_2 != model_circuit_3