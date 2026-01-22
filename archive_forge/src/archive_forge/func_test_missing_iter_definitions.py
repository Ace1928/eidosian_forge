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
def test_missing_iter_definitions():

    class FakeMissingIterSimulatorImpl(SimulatesAmplitudes, SimulatesExpectationValues, SimulatesFinalState):
        """A class which fails to define simulator methods."""
    missing_iter_sim = FakeMissingIterSimulatorImpl()
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0))
    bitstrings = [0]
    params = {}
    with pytest.raises(RecursionError):
        missing_iter_sim.compute_amplitudes_sweep(circuit, bitstrings, params)
    with pytest.raises(RecursionError):
        amp_iter = missing_iter_sim.compute_amplitudes_sweep_iter(circuit, bitstrings, params)
        next(amp_iter)
    obs = cirq.X(q0)
    with pytest.raises(RecursionError):
        missing_iter_sim.simulate_expectation_values_sweep(circuit, obs, params)
    with pytest.raises(RecursionError):
        ev_iter = missing_iter_sim.simulate_expectation_values_sweep_iter(circuit, obs, params)
        next(ev_iter)
    with pytest.raises(RecursionError):
        missing_iter_sim.simulate_sweep(circuit, params)
    with pytest.raises(RecursionError):
        state_iter = missing_iter_sim.simulate_sweep_iter(circuit, params)
        next(state_iter)