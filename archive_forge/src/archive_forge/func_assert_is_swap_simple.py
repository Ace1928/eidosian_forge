import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def assert_is_swap_simple(val: cirq.SupportsConsistentApplyUnitary) -> None:
    qid_shape = (2, 2)
    op_indices = [0, 1]
    state = np.arange(3 * 3, dtype=np.complex64).reshape((1, 3, 3))
    expected = state.copy()
    buf = expected[..., 0, 1].copy()
    expected[..., 0, 1] = expected[..., 1, 0]
    expected[..., 1, 0] = buf
    expected[..., :2, :2] *= 1j
    args = cirq.ApplyUnitaryArgs(state, np.empty_like(state), [1, 2])
    sub_args = args._for_operation_with_qid_shape(op_indices, tuple((qid_shape[i] for i in op_indices)))
    sub_result = val._apply_unitary_(sub_args)
    result = _incorporate_result_into_target(args, sub_args, sub_result)
    np.testing.assert_allclose(result, expected, atol=1e-08)