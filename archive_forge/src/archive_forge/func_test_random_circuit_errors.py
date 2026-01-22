from typing import Optional, Dict, Sequence, Union, cast
import random
import numpy as np
import pytest
import cirq
import cirq.testing
def test_random_circuit_errors():
    with pytest.raises(ValueError, match='but was -1'):
        _ = cirq.testing.random_circuit(qubits=5, n_moments=5, op_density=-1)
    with pytest.raises(ValueError, match='empty'):
        _ = cirq.testing.random_circuit(qubits=5, n_moments=5, op_density=0.5, gate_domain={})
    with pytest.raises(ValueError, match='At least one'):
        _ = cirq.testing.random_circuit(qubits=0, n_moments=5, op_density=0.5)
    with pytest.raises(ValueError, match='At least one'):
        _ = cirq.testing.random_circuit(qubits=(), n_moments=5, op_density=0.5)
    with pytest.raises(ValueError, match='After removing gates that act on less than 1 qubits, gate_domain had no gates'):
        _ = cirq.testing.random_circuit(qubits=1, n_moments=5, op_density=0.5, gate_domain={cirq.CNOT: 2})