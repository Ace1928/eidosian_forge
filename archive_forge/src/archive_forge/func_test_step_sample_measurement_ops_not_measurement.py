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
def test_step_sample_measurement_ops_not_measurement():
    q0 = cirq.LineQubit(0)
    step_result = FakeStepResult(ones_qubits=[q0])
    with pytest.raises(ValueError, match='MeasurementGate'):
        step_result.sample_measurement_ops([cirq.X(q0)])