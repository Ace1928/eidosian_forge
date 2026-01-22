import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_large_untangled_okay():
    circuit = cirq.Circuit()
    for i in range(59):
        for _ in range(9):
            circuit.append(cirq.X(cirq.LineQubit(i)))
        circuit.append(cirq.measure(cirq.LineQubit(i)))
    with pytest.raises(MemoryError, match='Unable to allocate'):
        _ = cirq.DensityMatrixSimulator(split_untangled_states=False).simulate(circuit)
    result = cirq.DensityMatrixSimulator().simulate(circuit)
    assert set(result._final_simulator_state.qubits) == set(cirq.LineQubit.range(59))
    result = cirq.DensityMatrixSimulator().run(circuit, repetitions=1000)
    assert len(result.measurements) == 59
    assert len(result.measurements['q(0)']) == 1000
    assert (result.measurements['q(0)'] == np.full(1000, 1)).all()