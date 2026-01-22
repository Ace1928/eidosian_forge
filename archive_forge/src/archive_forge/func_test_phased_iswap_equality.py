import numpy as np
import pytest
import scipy
import sympy
import cirq
def test_phased_iswap_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.PhasedISwapPowGate(phase_exponent=0, exponent=0.4), cirq.ISWAP ** 0.4)
    eq.add_equality_group(cirq.PhasedISwapPowGate(phase_exponent=0, exponent=0.4, global_shift=0.3), cirq.ISwapPowGate(global_shift=0.3) ** 0.4)