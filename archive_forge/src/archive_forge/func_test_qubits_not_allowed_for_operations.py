from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
def test_qubits_not_allowed_for_operations():

    class Op(cirq.Operation):

        @property
        def qubits(self) -> Tuple['cirq.Qid', ...]:
            pass

        def with_qubits(self, *new_qubits: 'cirq.Qid') -> Self:
            pass
    state = ExampleSimulationState()
    with pytest.raises(ValueError, match='Calls to act_on should not supply qubits if the action is an Operation'):
        cirq.act_on(Op(), state, qubits=[])