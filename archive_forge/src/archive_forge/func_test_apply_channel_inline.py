import numpy as np
import pytest
import cirq
def test_apply_channel_inline():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasApplyChannel:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            args.target_tensor = 0.5 * args.target_tensor + 0.5 * np.dot(np.dot(x, args.target_tensor), x)
            return args.target_tensor
    rho = np.copy(x)
    result = apply_channel(HasApplyChannel(), rho, [0], [1])
    np.testing.assert_almost_equal(result, x)