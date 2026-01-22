import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_assert_has_consistent_apply_unitary():

    class IdentityReturningUnalteredWorkspace:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            return args.available_buffer

        def _unitary_(self):
            return np.eye(2)

        def _num_qubits_(self):
            return 1
    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_unitary(IdentityReturningUnalteredWorkspace())

    class DifferentEffect:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            o = args.subspace_index(0)
            i = args.subspace_index(1)
            args.available_buffer[o] = args.target_tensor[i]
            args.available_buffer[i] = args.target_tensor[o]
            return args.available_buffer

        def _unitary_(self):
            return np.eye(2, dtype=np.complex128)

        def _num_qubits_(self):
            return 1
    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_unitary(DifferentEffect())

    class IgnoreAxisEffect:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            if args.target_tensor.shape[0] > 1:
                args.available_buffer[0] = args.target_tensor[1]
                args.available_buffer[1] = args.target_tensor[0]
            return args.available_buffer

        def _unitary_(self):
            return np.array([[0, 1], [1, 0]])

        def _num_qubits_(self):
            return 1
    with pytest.raises(AssertionError, match='Not equal|acted differently'):
        cirq.testing.assert_has_consistent_apply_unitary(IgnoreAxisEffect())

    class SameEffect:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            o = args.subspace_index(0)
            i = args.subspace_index(1)
            args.available_buffer[o] = args.target_tensor[i]
            args.available_buffer[i] = args.target_tensor[o]
            return args.available_buffer

        def _unitary_(self):
            return np.array([[0, 1], [1, 0]])

        def _num_qubits_(self):
            return 1
    cirq.testing.assert_has_consistent_apply_unitary(SameEffect())

    class SameQuditEffect:

        def _qid_shape_(self):
            return (3,)

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            args.available_buffer[..., 0] = args.target_tensor[..., 2]
            args.available_buffer[..., 1] = args.target_tensor[..., 0]
            args.available_buffer[..., 2] = args.target_tensor[..., 1]
            return args.available_buffer

        def _unitary_(self):
            return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    cirq.testing.assert_has_consistent_apply_unitary(SameQuditEffect())

    class BadExponent:

        def __init__(self, power):
            self.power = power

        def __pow__(self, power):
            return BadExponent(self.power * power)

        def _num_qubits_(self):
            return 1

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            i = args.subspace_index(1)
            args.target_tensor[i] *= self.power * 2
            return args.target_tensor

        def _unitary_(self):
            return np.array([[1, 0], [0, 2]])
    cirq.testing.assert_has_consistent_apply_unitary(BadExponent(1))
    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(BadExponent(1), exponents=[1, 2])

    class EffectWithoutUnitary:

        def _num_qubits_(self):
            return 1

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            return args.target_tensor
    cirq.testing.assert_has_consistent_apply_unitary(EffectWithoutUnitary())

    class NoEffect:

        def _num_qubits_(self):
            return 1

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            return NotImplemented
    cirq.testing.assert_has_consistent_apply_unitary(NoEffect())

    class UnknownCountEffect:
        pass
    with pytest.raises(TypeError, match='no _num_qubits_ or _qid_shape_'):
        cirq.testing.assert_has_consistent_apply_unitary(UnknownCountEffect())
    cirq.testing.assert_has_consistent_apply_unitary(cirq.X)
    cirq.testing.assert_has_consistent_apply_unitary(cirq.X.on(cirq.NamedQubit('q')))