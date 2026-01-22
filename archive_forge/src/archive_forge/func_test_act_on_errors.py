from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
def test_act_on_errors():

    class Op(cirq.Operation):

        @property
        def qubits(self) -> Tuple['cirq.Qid', ...]:
            pass

        def with_qubits(self, *new_qubits: 'cirq.Qid') -> Self:
            pass

        def _act_on_(self, sim_state):
            return False
    state = ExampleSimulationState(fallback_result=True)
    with pytest.raises(ValueError, match='_act_on_ must return True or NotImplemented'):
        cirq.act_on(Op(), state)