from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
def assert_test_circuit_for_dm_simulator(test_circuit, control_circuit) -> None:
    for split_untangled_states in [True, False]:
        sim = cirq.DensityMatrixSimulator(split_untangled_states=split_untangled_states)
        control_sim = sim.simulate(control_circuit).final_density_matrix
        test_sim = sim.simulate(test_circuit).final_density_matrix
        assert np.allclose(test_sim, control_sim)