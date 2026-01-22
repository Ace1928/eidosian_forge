import numpy as np
import pytest
import sympy
import cirq
def test_phased_fsim_approx_eq():
    assert cirq.approx_eq(cirq.PhasedFSimGate(1, 2, 3, 4, 5), cirq.PhasedFSimGate(1.00001, 2.00001, 3.00001, 4.00004, 5.00005), atol=0.01)