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
def test_floquet_get_calibrations():
    parameters_ab = cirq_google.PhasedFSimCharacterization(theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2)
    parameters_bc = cirq_google.PhasedFSimCharacterization(theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2)
    parameters_cd_dict = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    parameters_cd = cirq_google.PhasedFSimCharacterization(**parameters_cd_dict)
    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(parameters={(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd_dict})
    requests = [_create_sqrt_iswap_request([(a, b), (c, d)]), _create_sqrt_iswap_request([(b, c)])]
    results = engine_simulator.get_calibrations(requests)
    assert results == [cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(np.pi / 4, 0.0), parameters={(a, b): parameters_ab, (c, d): parameters_cd}, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION), cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(np.pi / 4, 0.0), parameters={(b, c): parameters_bc}, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)]