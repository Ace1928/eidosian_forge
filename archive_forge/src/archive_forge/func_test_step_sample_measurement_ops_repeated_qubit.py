import abc
from typing import Generic, Dict, Any, List, Sequence, Union
from unittest import mock
import duet
import numpy as np
import pytest
import cirq
from cirq import study
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulator import (
def test_step_sample_measurement_ops_repeated_qubit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    step_result = FakeStepResult(ones_qubits=[q0])
    with pytest.raises(ValueError, match='Measurement key q\\(0\\) repeated'):
        step_result.sample_measurement_ops([cirq.measure(q0), cirq.measure(q1, q2), cirq.measure(q0)])