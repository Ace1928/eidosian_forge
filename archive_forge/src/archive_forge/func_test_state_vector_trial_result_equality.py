import numpy as np
import cirq
import cirq.testing
def test_state_vector_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    final_simulator_state = cirq.StateVectorSimulationState(initial_state=np.array([]))
    eq.add_equality_group(cirq.StateVectorTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state), cirq.StateVectorTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=final_simulator_state))
    eq.add_equality_group(cirq.StateVectorTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={}, final_simulator_state=final_simulator_state))
    eq.add_equality_group(cirq.StateVectorTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state))
    final_simulator_state = cirq.StateVectorSimulationState(initial_state=np.array([1]))
    eq.add_equality_group(cirq.StateVectorTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state))