from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_inline():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasApplyMixture:

        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            args.target_tensor = 0.5 * args.target_tensor + 0.5 * np.dot(np.dot(x, args.target_tensor), x)
            return args.target_tensor
    rho = np.copy(x)
    assert_apply_mixture_returns(HasApplyMixture(), rho, [0], [1], expected_result=x)