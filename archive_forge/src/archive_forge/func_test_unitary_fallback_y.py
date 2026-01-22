import numpy as np
import pytest
import cirq
def test_unitary_fallback_y():

    class UnitaryYGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

        def _unitary_(self):
            return np.array([[0, -1j], [1j, 0]])
    args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    cirq.act_on(UnitaryYGate(), args, [cirq.LineQubit(1)])
    expected_args = cirq.StabilizerChFormSimulationState(qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    cirq.act_on(cirq.Y, expected_args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(args.state.state_vector(), expected_args.state.state_vector())