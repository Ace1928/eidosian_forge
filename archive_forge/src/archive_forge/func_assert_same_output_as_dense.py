import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def assert_same_output_as_dense(circuit, qubit_order, initial_state=0, grouping=None):
    mps_simulator = ccq.mps_simulator.MPSSimulator(grouping=grouping)
    ref_simulator = cirq.Simulator()
    actual = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    expected = ref_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    np.testing.assert_allclose(actual.final_state.to_numpy(), expected.final_state_vector, atol=0.0001)
    assert len(actual.measurements) == 0