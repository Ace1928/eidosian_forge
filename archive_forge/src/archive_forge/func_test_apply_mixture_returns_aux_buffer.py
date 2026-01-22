from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_returns_aux_buffer():
    rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)

    class ReturnsAuxBuffer0:

        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            return args.auxiliary_buffer0
    with pytest.raises(AssertionError, match='ReturnsAuxBuffer0'):
        assert_apply_mixture_returns(ReturnsAuxBuffer0(), rho, [0], [1])

    class ReturnsAuxBuffer1:

        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            return args.auxiliary_buffer1
    with pytest.raises(AssertionError, match='ReturnsAuxBuffer1'):
        assert_apply_mixture_returns(ReturnsAuxBuffer1(), rho, [0], [1])