import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_trial_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.StabilizerChFormSimulationState(qubits=[q0])
    result = cirq.CliffordTrialResult(params=cirq.ParamResolver({}), measurements={'m': np.array([[1]])}, final_simulator_state=final_simulator_state)
    cirq.testing.assert_repr_pretty(result, 'measurements: m=1\noutput state: |0⟩')
    cirq.testing.assert_repr_pretty(result, 'cirq.CliffordTrialResult(...)', cycle=True)