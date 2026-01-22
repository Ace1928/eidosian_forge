from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_no_protocols_implemented():

    class NoProtocols:
        pass
    rho = np.ones((2, 2, 2, 2), dtype=np.complex128)
    with pytest.raises(TypeError, match='has no'):
        assert_apply_mixture_returns(NoProtocols(), rho, left_axes=[1], right_axes=[1])