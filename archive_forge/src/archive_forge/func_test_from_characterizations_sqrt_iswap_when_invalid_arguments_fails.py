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
def test_from_characterizations_sqrt_iswap_when_invalid_arguments_fails():
    parameters_ab = cirq_google.PhasedFSimCharacterization(theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2)
    parameters_bc = cirq_google.PhasedFSimCharacterization(theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2)
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='multiple moments'):
        PhasedFSimEngineSimulator.create_from_characterizations_sqrt_iswap(characterizations=[cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(np.pi / 4, 0.0), parameters={(a, b): parameters_ab}, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION), cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(np.pi / 4, 0.0), parameters={(a, b): parameters_bc}, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)])
    with pytest.raises(AssertionError, match='Expected ISWA'):
        PhasedFSimEngineSimulator.create_from_characterizations_sqrt_iswap(characterizations=[cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(np.pi / 4, 0.2), parameters={(a, b): parameters_ab}, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)])
    with pytest.raises(ValueError, match='unparameterized'):
        PhasedFSimEngineSimulator.create_from_characterizations_sqrt_iswap(characterizations=[cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(np.pi / 4, sympy.Symbol('a')), parameters={(a, b): parameters_ab}, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)])
    with pytest.raises(AssertionError, match='Expected FSimGate'):
        PhasedFSimEngineSimulator.create_from_characterizations_sqrt_iswap(characterizations=[cirq_google.PhasedFSimCalibrationResult(gate=cirq.CNOT, parameters={(a, b): parameters_ab}, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)])