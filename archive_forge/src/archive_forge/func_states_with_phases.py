import copy
import pytest
import numpy as np
import cirq
def states_with_phases(st: np.ndarray):
    """Returns several states similar to st with modified global phases."""
    st = np.array(st, dtype='complex64')
    yield st
    phases = [np.exp(1j * np.pi / 6), -1j, 1j, -1, np.exp(-1j * np.pi / 28)]
    random = np.random.RandomState(1)
    for _ in range(3):
        curr_st = copy.deepcopy(st)
        cirq.to_valid_state_vector(curr_st, num_qubits=2)
        for i in range(4):
            phase = random.choice(phases)
            curr_st[i] *= phase
        yield curr_st