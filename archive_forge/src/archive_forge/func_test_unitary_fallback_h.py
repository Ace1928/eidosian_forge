import numpy as np
import pytest
import cirq
def test_unitary_fallback_h():

    class UnitaryHGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

        def _unitary_(self):
            return np.array([[1, 1], [1, -1]]) / 2 ** 0.5
    args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    cirq.act_on(UnitaryHGate(), args, [cirq.LineQubit(1)])
    expected_args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    cirq.act_on(cirq.H, expected_args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(args.state.state_vector(), expected_args.state.state_vector())