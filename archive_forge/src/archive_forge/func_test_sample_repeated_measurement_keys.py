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
def test_sample_repeated_measurement_keys():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append([cirq.measure(q[0], key='a'), cirq.measure(q[1], key='a'), cirq.measure(q[0], key='b'), cirq.measure(q[1], key='b')])
    result = cirq.sample(circuit)
    assert len(result.records['a']) == 1
    assert len(result.records['b']) == 1
    assert len(result.records['a'][0]) == 2
    assert len(result.records['b'][0]) == 2