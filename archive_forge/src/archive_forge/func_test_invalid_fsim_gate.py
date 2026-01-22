from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('bad_fsim_gate', (cirq.FSimGate(theta=0, phi=0), cirq.FSimGate(theta=np.pi / 4, phi=np.pi / 2)))
def test_invalid_fsim_gate(bad_fsim_gate):
    with pytest.raises(ValueError):
        cirq.decompose_cphase_into_two_fsim(cirq.CZ, fsim_gate=bad_fsim_gate)