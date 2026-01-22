from typing import List, Union
import pytest
import sympy
import numpy as np
import cirq
import cirq_google
@pytest.mark.parametrize('gate, params, target_type', [*[(g, param, cirq.IdentityGate) for g, param in VALID_IDENTITY], *[(g, param, cirq.CZPowGate) for g, param in VALID_CZPOW_GATES], *[(g, param, cirq.ISwapPowGate) for g, param in VALID_ISWAP_GATES], *[(g, param, cirq.PhasedISwapPowGate) for g, param in VALID_PHASED_ISWAP_GATES], *[(g, param, cirq.FSimGate) for g, param in VALID_FSIM_GATES], *[(g, param, cirq.PhasedFSimGate) for g, param in VALID_PHASED_FSIM_GATES]])
def test_fsim_gate_family_convert_accept(gate, params, target_type):
    gate_family_allow_symbols = cirq_google.FSimGateFamily(allow_symbols=True)
    assert isinstance(gate_family_allow_symbols.convert(gate, target_type), target_type)
    resolved_gate = cirq.resolve_parameters(gate, params)
    target_gate = cirq_google.FSimGateFamily().convert(resolved_gate, target_type)
    assert isinstance(target_gate, target_type)
    np.testing.assert_array_almost_equal(cirq.unitary(resolved_gate), cirq.unitary(target_gate))
    assert gate in cirq_google.FSimGateFamily(gates_to_accept=[target_type], allow_symbols=True)
    assert gate in cirq_google.FSimGateFamily(gates_to_accept=[resolved_gate], allow_symbols=True)
    assert resolved_gate in cirq_google.FSimGateFamily(gates_to_accept=[target_type])
    assert resolved_gate in cirq_google.FSimGateFamily(gates_to_accept=[resolved_gate])