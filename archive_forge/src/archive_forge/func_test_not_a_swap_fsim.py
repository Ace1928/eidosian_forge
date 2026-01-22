import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
@pytest.mark.parametrize('theta', (0, 5 * np.pi, -np.pi))
def test_not_a_swap_fsim(theta):
    a, b = cirq.LineQubit.range(2)
    assert not _is_swaplike(cirq.FSimGate(theta=theta, phi=0.456).on(a, b))