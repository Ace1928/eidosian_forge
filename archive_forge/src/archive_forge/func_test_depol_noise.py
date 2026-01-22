import cirq
import cirq.contrib.noise_models as ccn
from cirq import ops
from cirq.testing import assert_equivalent_op_tree
def test_depol_noise():
    noise_model = ccn.DepolarizingNoiseModel(depol_prob=0.005)
    qubits = cirq.LineQubit.range(2)
    moment = cirq.Moment([cirq.X(qubits[0]), cirq.Y(qubits[1])])
    noisy_mom = noise_model.noisy_moment(moment, system_qubits=qubits)
    assert len(noisy_mom) == 2
    assert noisy_mom[0] == moment
    for g in noisy_mom[1]:
        assert isinstance(g.gate, cirq.DepolarizingChannel)