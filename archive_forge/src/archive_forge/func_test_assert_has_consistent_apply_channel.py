import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_assert_has_consistent_apply_channel():

    class Correct:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            args.target_tensor[...] = 0
            return args.target_tensor

        def _kraus_(self):
            return [np.array([[0, 0], [0, 0]])]

        def _num_qubits_(self):
            return 1
    cirq.testing.assert_has_consistent_apply_channel(Correct())

    class Wrong:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            args.target_tensor[...] = 0
            return args.target_tensor

        def _kraus_(self):
            return [np.array([[1, 0], [0, 0]])]

        def _num_qubits_(self):
            return 1
    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_channel(Wrong())

    class NoNothing:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            return NotImplemented

        def _kraus_(self):
            return NotImplemented

        def _num_qubits_(self):
            return 1
    cirq.testing.assert_has_consistent_apply_channel(NoNothing())

    class NoKraus:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            return args.target_tensor

        def _kraus_(self):
            return NotImplemented

        def _num_qubits_(self):
            return 1
    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_channel(NoKraus())

    class NoApply:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            return NotImplemented

        def _kraus_(self):
            return [np.array([[0, 0], [0, 0]])]

        def _num_qubits_(self):
            return 1
    cirq.testing.assert_has_consistent_apply_channel(NoApply())