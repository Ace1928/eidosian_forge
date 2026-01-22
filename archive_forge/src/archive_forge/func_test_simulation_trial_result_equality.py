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
def test_simulation_trial_result_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.SimulationTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=()), cirq.SimulationTrialResult(params=cirq.ParamResolver({}), measurements={}, final_simulator_state=()))
    eq.add_equality_group(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={}, final_simulator_state=()))
    eq.add_equality_group(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([1])}, final_simulator_state=()))
    eq.add_equality_group(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([1])}, final_simulator_state=(0, 1)))