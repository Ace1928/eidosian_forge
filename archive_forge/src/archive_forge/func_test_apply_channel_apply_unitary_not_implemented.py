import numpy as np
import pytest
import cirq
def test_apply_channel_apply_unitary_not_implemented():

    class ApplyUnitaryNotImplemented:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return NotImplemented
    rho = np.ones((2, 2, 2, 2), dtype=np.complex128)
    out_buf, aux_buf0, aux_buf1 = make_buffers((2, 2, 2, 2), dtype=rho.dtype)
    with pytest.raises(TypeError):
        cirq.apply_channel(ApplyUnitaryNotImplemented(), args=cirq.ApplyChannelArgs(target_tensor=rho, left_axes=[1], right_axes=[3], out_buffer=out_buf, auxiliary_buffer0=aux_buf0, auxiliary_buffer1=aux_buf1))