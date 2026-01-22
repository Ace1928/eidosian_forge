import numpy as np
import pytest
import cirq
def test_apply_channel_no_protocols_implemented():

    class NoProtocols:
        pass
    rho = np.ones((2, 2, 2, 2), dtype=np.complex128)
    with pytest.raises(TypeError):
        apply_channel(NoProtocols(), rho, left_axes=[1], right_axes=[1])