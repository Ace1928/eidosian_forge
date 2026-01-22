import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_trial_result_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.StabilizerChFormSimulationState(qubits=[q0])
    assert str(cirq.CliffordTrialResult(params=cirq.ParamResolver({}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state)) == 'measurements: m=1\noutput state: |0⟩'