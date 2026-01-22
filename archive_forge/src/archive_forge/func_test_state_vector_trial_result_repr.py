import numpy as np
import cirq
import cirq.testing
def test_state_vector_trial_result_repr():
    q0 = cirq.NamedQubit('a')
    final_simulator_state = cirq.StateVectorSimulationState(available_buffer=np.array([0, 1], dtype=np.complex64), prng=np.random.RandomState(0), qubits=[q0], initial_state=np.array([0, 1], dtype=np.complex64), dtype=np.complex64)
    trial_result = cirq.StateVectorTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([[1]], dtype=np.int32)}, final_simulator_state=final_simulator_state)
    expected_repr = "cirq.StateVectorTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([[1]], dtype=np.dtype('int32'))}, final_simulator_state=cirq.StateVectorSimulationState(initial_state=np.array([0j, (1+0j)], dtype=np.dtype('complex64')), qubits=(cirq.NamedQubit('a'),), classical_data=cirq.ClassicalDataDictionaryStore()))"
    assert repr(trial_result) == expected_repr
    assert eval(expected_repr) == trial_result