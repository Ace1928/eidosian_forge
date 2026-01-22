import numpy as np
import cirq
import cirq.testing
def test_state_vector_trial_result_qid_shape():
    final_simulator_state = cirq.StateVectorSimulationState(qubits=[cirq.NamedQubit('a')], initial_state=np.array([0, 1]))
    trial_result = cirq.StateVectorTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state)
    assert cirq.qid_shape(trial_result) == (2,)
    final_simulator_state = cirq.StateVectorSimulationState(qubits=cirq.LineQid.for_qid_shape((3, 2)), initial_state=np.array([0, 0, 0, 0, 1, 0]))
    trial_result = cirq.StateVectorTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([[2, 0]])}, final_simulator_state=final_simulator_state)
    assert cirq.qid_shape(trial_result) == (3, 2)