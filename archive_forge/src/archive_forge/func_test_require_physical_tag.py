import cirq
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG, OpIdentifier
def test_require_physical_tag():
    q0, q1 = cirq.LineQubit.range(2)
    op_id0 = OpIdentifier(cirq.XPowGate, q0)
    op_id1 = OpIdentifier(cirq.ZPowGate, q1)
    model = InsertionNoiseModel({op_id0: cirq.T(q0), op_id1: cirq.H(q1)})
    assert model.require_physical_tag
    moment_0 = cirq.Moment(cirq.X(q0).with_tags(PHYSICAL_GATE_TAG), cirq.Z(q1))
    assert model.noisy_moment(moment_0, system_qubits=[q0, q1]) == [moment_0, cirq.Moment(cirq.T(q0))]