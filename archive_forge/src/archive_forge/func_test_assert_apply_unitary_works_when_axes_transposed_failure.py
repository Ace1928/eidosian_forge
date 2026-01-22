import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_assert_apply_unitary_works_when_axes_transposed_failure():

    class BadOp:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            a, b = args.axes
            rest = list(range(len(args.target_tensor.shape)))
            rest.remove(a)
            rest.remove(b)
            size = args.target_tensor.size
            view = args.target_tensor.transpose([a, b, *rest])
            view = view.reshape((4, size // 4))
            view[1, ...] *= 1j
            view[2, ...] *= -1
            view[3, ...] *= -1j
            return args.target_tensor

        def _num_qubits_(self):
            return 2
    bad_op = BadOp()
    assert cirq.has_unitary(bad_op)
    np.testing.assert_allclose(cirq.unitary(bad_op), np.diag([1, 1j, -1, -1j]))
    with pytest.raises(AssertionError, match='acted differently on out-of-order axes'):
        for _ in range(100):
            _assert_apply_unitary_works_when_axes_transposed(bad_op)