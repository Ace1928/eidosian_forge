from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
def test_qubits_should_be_defined_for_operations():
    state = ExampleSimulationState()
    with pytest.raises(ValueError, match='Calls to act_on should'):
        cirq.act_on(cirq.KrausChannel([np.array([[1, 0], [0, 0]])]), state, qubits=None)