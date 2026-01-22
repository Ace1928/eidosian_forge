from typing import Any, Sequence
import numpy as np
import pytest
import cirq
from cirq.sim import simulation_state
from cirq.testing import PhaseUsingCleanAncilla, PhaseUsingDirtyAncilla
def test_measurements():
    args = ExampleSimulationState()
    args.measure([cirq.LineQubit(0)], 'test', [False], {})
    assert args.log_of_measurement_results['test'] == [5]