import numpy as np
import cirq
import cirq.testing
def test_str_qudit():
    qutrit = cirq.LineQid(0, dimension=3)
    final_simulator_state = cirq.StateVectorSimulationState(prng=np.random.RandomState(0), qubits=[qutrit], initial_state=np.array([0, 0, 1]), dtype=np.complex64)
    result = cirq.StateVectorTrialResult(cirq.ParamResolver(), {}, final_simulator_state)
    assert '|2⟩' in str(result)
    ququart = cirq.LineQid(0, dimension=4)
    final_simulator_state = cirq.StateVectorSimulationState(prng=np.random.RandomState(0), qubits=[ququart], initial_state=np.array([0, 1, 0, 0]), dtype=np.complex64)
    result = cirq.StateVectorTrialResult(cirq.ParamResolver(), {}, final_simulator_state)
    assert '|1⟩' in str(result)