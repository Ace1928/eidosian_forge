from typing import Tuple
import numpy as np
import pytest
import cirq
def test_cannot_act():

    class NoDetails:
        pass

    class NoDetailsSingleQubitGate(cirq.testing.SingleQubitGate):
        pass
    args = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=3), qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    with pytest.raises(TypeError, match='no _num_qubits_ or _qid_shape_'):
        cirq.act_on(NoDetails(), args, [cirq.LineQubit(1)])
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(NoDetailsSingleQubitGate(), args, [cirq.LineQubit(1)])