import numpy as np
import pytest
import cirq
import cirq_google
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse
def test_sycamore_grid_layout():
    q0 = cirq.GridQubit(5, 5)
    q1 = cirq.GridQubit(5, 6)
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)
    cirq_google.Sycamore.validate_operation(syc)
    cirq_google.Sycamore.validate_operation(sqrt_iswap)
    with pytest.raises(ValueError):
        cirq_google.Sycamore23.validate_operation(syc)
    with pytest.raises(ValueError):
        cirq_google.Sycamore23.validate_operation(sqrt_iswap)