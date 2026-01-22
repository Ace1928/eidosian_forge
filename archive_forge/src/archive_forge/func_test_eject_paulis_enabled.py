import pytest
import cirq
import cirq_google
@pytest.mark.parametrize('before, expected, gate_family', [(cirq.Circuit(cirq.Z(_qa) ** 0.75, cirq.CZ(_qa, _qb)), cirq.Circuit(cirq.CZ(_qa, _qb), cirq.Z(_qa) ** 0.75), cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()])), (cirq.Circuit((cirq.Z ** 0.75)(_qa).with_tags(cirq_google.PhysicalZTag()), cirq.CZ(_qa, _qb)), cirq.Circuit(cirq.CZ(_qa, _qb), cirq.Z(_qa) ** 0.75), cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()])), (cirq.Circuit(cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa), cirq.CZ(_qa, _qb)), cirq.Circuit((cirq.CZ ** (-1))(_qa, _qb), cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa), cirq.Z(_qb)), cirq.PhasedXPowGate)])
def test_eject_paulis_enabled(before, expected, gate_family):
    after = cirq.optimize_for_target_gateset(before, gateset=cirq_google.GoogleCZTargetGateset(eject_paulis=True, additional_gates=[gate_family]), ignore_failures=False)
    cirq.testing.assert_same_circuits(after, expected)