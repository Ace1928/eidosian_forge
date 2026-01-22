from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_invalid_qubits():
    with pytest.raises(ValueError):
        cirq.decompose_cphase_into_two_fsim(cphase_gate=cirq.CZ, fsim_gate=FakeSycamoreGate(), qubits=cirq.LineQubit.range(3))