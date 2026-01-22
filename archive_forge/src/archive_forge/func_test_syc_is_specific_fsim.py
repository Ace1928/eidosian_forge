import numpy as np
import pytest
import cirq
import cirq_google as cg
def test_syc_is_specific_fsim():
    assert cg.SYC == cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)