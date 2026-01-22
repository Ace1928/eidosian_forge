from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_decomposed_fallback():

    class Composite(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)
    args = cirq.StateVectorSimulationState(available_buffer=np.empty((2, 2, 2), dtype=np.complex64), qubits=cirq.LineQubit.range(3), prng=np.random.RandomState(), initial_state=cirq.one_hot(shape=(2, 2, 2), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(Composite(), args, [cirq.LineQubit(1)])
    np.testing.assert_allclose(args.target_tensor, cirq.one_hot(index=(0, 1, 0), shape=(2, 2, 2), dtype=np.complex64))