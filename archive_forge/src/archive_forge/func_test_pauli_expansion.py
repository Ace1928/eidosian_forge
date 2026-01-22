import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('val, expected_expansion', ((ReturnsExpansion(cirq.LinearDict({'X': 1, 'Y': 2, 'Z': 3})), cirq.LinearDict({'X': 1, 'Y': 2, 'Z': 3})), (HasUnitary(np.eye(2)), cirq.LinearDict({'I': 1})), (HasUnitary(np.array([[1, -1j], [1j, -1]])), cirq.LinearDict({'Y': 1, 'Z': 1})), (HasUnitary(np.array([[0.0, 1.0], [0.0, 0.0]])), cirq.LinearDict({'X': 0.5, 'Y': 0.5j})), (HasUnitary(np.eye(16)), cirq.LinearDict({'IIII': 1.0})), (cirq.H, cirq.LinearDict({'X': np.sqrt(0.5), 'Z': np.sqrt(0.5)})), (cirq.ry(np.pi / 2), cirq.LinearDict({'I': np.cos(np.pi / 4), 'Y': -1j * np.sin(np.pi / 4)}))))
def test_pauli_expansion(val, expected_expansion):
    actual_expansion = cirq.pauli_expansion(val)
    assert cirq.approx_eq(actual_expansion, expected_expansion, atol=1e-12)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert np.abs(actual_expansion[name] - expected_expansion[name]) < 1e-12