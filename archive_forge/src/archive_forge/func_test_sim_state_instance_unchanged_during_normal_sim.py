import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('split', [True, False])
def test_sim_state_instance_unchanged_during_normal_sim(split: bool):
    sim = SplittableCountingSimulator(split_untangled_states=split)
    state = sim._create_simulation_state(0, (q0, q1))
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.reset(q1))
    for step in sim.simulate_moment_steps(circuit, initial_state=state):
        assert step._sim_state is state
        assert (step._merged_sim_state is not state) == split