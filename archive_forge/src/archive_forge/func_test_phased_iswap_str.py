import numpy as np
import pytest
import scipy
import sympy
import cirq
def test_phased_iswap_str():
    assert str(cirq.PhasedISwapPowGate(exponent=1)) == 'PhasedISWAP'
    assert str(cirq.PhasedISwapPowGate(exponent=0.5)) == 'PhasedISWAP**0.5'
    assert str(cirq.PhasedISwapPowGate(exponent=0.5, global_shift=0.5)) == 'PhasedISWAP(exponent=0.5, global_shift=0.5)'