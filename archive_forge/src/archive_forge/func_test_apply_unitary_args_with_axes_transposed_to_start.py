import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_apply_unitary_args_with_axes_transposed_to_start():
    target = np.zeros((2, 3, 4, 5))
    buffer = np.zeros((2, 3, 4, 5))
    args = cirq.ApplyUnitaryArgs(target, buffer, [1, 3])
    new_args = args.with_axes_transposed_to_start()
    assert new_args.target_tensor.shape == (3, 5, 2, 4)
    assert new_args.available_buffer.shape == (3, 5, 2, 4)
    assert new_args.axes == (0, 1)
    new_args.target_tensor[2, 4, 1, 3] = 1
    assert args.target_tensor[1, 2, 3, 4] == 1
    new_args.available_buffer[2, 4, 1, 3] = 2
    assert args.available_buffer[1, 2, 3, 4] == 2