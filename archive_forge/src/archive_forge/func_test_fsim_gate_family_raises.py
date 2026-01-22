from typing import List, Union
import pytest
import sympy
import numpy as np
import cirq
import cirq_google
def test_fsim_gate_family_raises():
    with pytest.raises(ValueError, match='must be one of'):
        _ = cirq_google.FSimGateFamily(gate_types_to_check=[cirq_google.SycamoreGate])
    with pytest.raises(ValueError, match='Parameterized gate'):
        _ = cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ ** sympy.Symbol('THETA')])
    with pytest.raises(ValueError, match='must be either a type from or an instance of'):
        _ = cirq_google.FSimGateFamily(gates_to_accept=[cirq.CNOT])
    with pytest.raises(ValueError, match='must be either a type from or an instance of'):
        _ = cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SycamoreGate])
    with pytest.raises(ValueError, match='must be one of'):
        _ = cirq_google.FSimGateFamily().convert(cirq.ISWAP, cirq_google.SycamoreGate)