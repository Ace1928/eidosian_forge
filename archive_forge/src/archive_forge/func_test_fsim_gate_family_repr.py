from typing import List, Union
import pytest
import sympy
import numpy as np
import cirq
import cirq_google
@pytest.mark.parametrize('gate_family', [cirq_google.FSimGateFamily(), cirq_google.FSimGateFamily(allow_symbols=True), cirq_google.FSimGateFamily(gates_to_accept=[cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6), cirq.SQRT_ISWAP, cirq.CZPowGate, cirq.PhasedISwapPowGate], gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1] + [cirq.FSimGate], allow_symbols=True, atol=1e-08)])
def test_fsim_gate_family_repr(gate_family):
    cirq.testing.assert_equivalent_repr(gate_family, setup_code='import cirq\nimport cirq_google')
    assert 'FSimGateFamily' in str(gate_family)