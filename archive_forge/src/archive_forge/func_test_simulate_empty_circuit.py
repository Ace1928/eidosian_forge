import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_simulate_empty_circuit():
    sim = CountingSimulator()
    r = sim.simulate(cirq.Circuit())
    assert r._final_simulator_state.gate_count == 0
    assert r._final_simulator_state.measurement_count == 0
    assert r._final_simulator_state.copy_count == 0