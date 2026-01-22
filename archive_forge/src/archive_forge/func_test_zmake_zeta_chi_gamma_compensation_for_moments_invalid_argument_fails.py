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
def test_zmake_zeta_chi_gamma_compensation_for_moments_invalid_argument_fails() -> None:
    a, b, c = cirq.LineQubit.range(3)
    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit(), [1])
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])
    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit(SQRT_ISWAP_INV_GATE.on(a, b)), [None])
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])
    with pytest.raises(ValueError):
        workflow.make_zeta_chi_gamma_compensation_for_moments(cirq.Circuit(SQRT_ISWAP_INV_GATE.on(a, b)), [])
    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit(SQRT_ISWAP_INV_GATE.on(a, b)), [0])
        characterizations = [PhasedFSimCalibrationResult(parameters={}, gate=SQRT_ISWAP_INV_GATE, options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)]
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, characterizations)
    with pytest.raises(workflow.IncompatibleMomentError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit(cirq.global_phase_operation(coefficient=1.0)), [None])
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])
    with pytest.raises(workflow.IncompatibleMomentError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit(cirq.CX.on(a, b)), [None])
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])
    with pytest.raises(workflow.IncompatibleMomentError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit([SQRT_ISWAP_INV_GATE.on(a, b), cirq.Z.on(c)]), [0])
        characterizations = [PhasedFSimCalibrationResult(parameters={(a, b): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5)}, gate=SQRT_ISWAP_INV_GATE, options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)]
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, characterizations)