import numpy as np
import cirq
import cirq.testing
def test_state_vector_trial_result_no_qubits():
    initial_state_vector = np.array([1], dtype=np.complex64)
    initial_state = initial_state_vector.reshape((2,) * 0)
    final_simulator_state = cirq.StateVectorSimulationState(qubits=[], initial_state=initial_state)
    trial_result = cirq.StateVectorTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state)
    state_vector = trial_result.state_vector()
    assert state_vector.shape == (1,)
    assert np.array_equal(state_vector, initial_state_vector)