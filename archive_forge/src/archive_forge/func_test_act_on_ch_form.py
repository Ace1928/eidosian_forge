import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_act_on_ch_form(phase):
    state = cirq.StabilizerStateChForm(0)
    args = cirq.StabilizerChFormSimulationState(qubits=[], prng=np.random.RandomState(), initial_state=state)
    cirq.act_on(cirq.global_phase_operation(phase), args, allow_decompose=False)
    assert state.state_vector() == [[phase]]