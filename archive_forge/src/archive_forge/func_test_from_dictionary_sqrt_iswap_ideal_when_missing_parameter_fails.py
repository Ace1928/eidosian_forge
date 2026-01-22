from typing import Iterable, Optional, Tuple
import collections
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq_google
from cirq_google.calibration.engine_simulator import (
from cirq_google.calibration import (
import cirq
def test_from_dictionary_sqrt_iswap_ideal_when_missing_parameter_fails():
    parameters_ab = cirq_google.PhasedFSimCharacterization(theta=0.8, zeta=-0.5, chi=-0.4)
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.FSimGate(np.pi / 4, 0.0).on(a, b))
    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(parameters={(a, b): parameters_ab})
    with pytest.raises(ValueError):
        engine_simulator.final_state_vector(circuit)