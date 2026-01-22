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
def test_run_zeta_chi_gamma_calibration_for_moments_no_chi() -> None:
    parameters_ab = cirq_google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.5, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=-0.5, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.2, gamma=0.4)
    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(parameters={(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd}, ideal_when_missing_parameter=True)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)], [SQRT_ISWAP_INV_GATE.on(b, c)]])
    calibrated_circuit, *_ = workflow.run_zeta_chi_gamma_compensation_for_moments(circuit, engine_simulator, processor_id=None)
    assert cirq.allclose_up_to_global_phase(engine_simulator.final_state_vector(calibrated_circuit.circuit), cirq.final_state_vector(circuit))