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
def test_floquet_get_calibrations_when_invalid_request_fails():
    parameters_ab = cirq_google.PhasedFSimCharacterization(theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2)
    a, b = cirq.LineQubit.range(2)
    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(parameters={(a, b): parameters_ab})
    with pytest.raises(ValueError):
        engine_simulator.get_calibrations([FloquetPhasedFSimCalibrationRequest(gate=cirq.FSimGate(np.pi / 4, 0.5), pairs=((a, b),), options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)])
    with pytest.raises(ValueError):
        engine_simulator.get_calibrations([ExamplePhasedFSimCalibrationRequest(gate=cirq.FSimGate(np.pi / 4, 0.5), pairs=((a, b),), options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)])