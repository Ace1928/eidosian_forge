import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_act_on():
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.act_on(cirq.ResetChannel(), ExampleSimulationState(), qubits=())
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(shape=(2, 2, 2, 2, 2), dtype=np.complex64), qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=cirq.one_hot(index=(1, 1, 1, 1, 1), shape=(2, 2, 2, 2, 2), dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(cirq.ResetChannel(), args, [cirq.LineQubit(1)])
    assert args.log_of_measurement_results == {}
    np.testing.assert_allclose(args.target_tensor, cirq.one_hot(index=(1, 0, 1, 1, 1), shape=(2, 2, 2, 2, 2), dtype=np.complex64))
    cirq.act_on(cirq.ResetChannel(), args, [cirq.LineQubit(1)])
    assert args.log_of_measurement_results == {}
    np.testing.assert_allclose(args.target_tensor, cirq.one_hot(index=(1, 0, 1, 1, 1), shape=(2, 2, 2, 2, 2), dtype=np.complex64))