import itertools
from typing import Optional
from unittest import mock
import numpy as np
import pytest
import cirq
import cirq_google
import cirq_google.calibration.workflow as workflow
import cirq_google.calibration.xeb_wrapper
from cirq.experiments import (
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
def test_make_zeta_chi_gamma_compensation_for_operations_with_permit_mixed_moments_disabled():
    a, b, c, d = cirq.LineQubit.range(4)
    parameters_ab = cirq_google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)
    parameters_dict = {(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd}
    circuit = cirq.Circuit([cirq.Moment([cirq.X(a), cirq.Y(c)]), cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)]), cirq.Moment([cirq.X(a), SQRT_ISWAP_INV_GATE.on(b, c), cirq.Y(d)])])
    options = cirq_google.FloquetPhasedFSimCalibrationOptions(characterize_theta=False, characterize_zeta=True, characterize_chi=True, characterize_gamma=True, characterize_phi=False)
    characterizations = [PhasedFSimCalibrationResult(parameters={pair: parameters}, gate=SQRT_ISWAP_INV_GATE, options=options) for pair, parameters in parameters_dict.items()]
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.make_zeta_chi_gamma_compensation_for_operations(circuit, characterizations)