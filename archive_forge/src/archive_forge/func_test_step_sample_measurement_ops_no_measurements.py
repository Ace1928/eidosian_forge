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
def test_step_sample_measurement_ops_no_measurements():
    step_result = FakeStepResult(ones_qubits=[])
    measurements = step_result.sample_measurement_ops([])
    assert measurements == {}