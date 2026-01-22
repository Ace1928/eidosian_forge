import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
@pytest.mark.parametrize('phase_exponent', np.linspace(0, 1, 5))
def test_phase_calibrated_fsim_gate_as_characterized_phased_fsim_gate(phase_exponent: float):
    a, b = cirq.LineQubit.range(2)
    ideal_gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    characterized_gate = cirq.PhasedFSimGate(theta=ideal_gate.theta, zeta=0.1, chi=0.2, gamma=0.3, phi=ideal_gate.phi)
    parameters = PhasedFSimCharacterization(theta=cast(float, ideal_gate.theta), zeta=cast(float, characterized_gate.zeta), chi=cast(float, characterized_gate.chi), gamma=cast(float, characterized_gate.gamma), phi=cast(float, ideal_gate.phi))
    calibrated = PhaseCalibratedFSimGate(ideal_gate, phase_exponent=phase_exponent)
    phased_gate = calibrated.as_characterized_phased_fsim_gate(parameters).on(a, b)
    assert np.allclose(cirq.unitary(phased_gate), cirq.unitary(cirq.Circuit([[cirq.Z(a) ** (-phase_exponent), cirq.Z(b) ** phase_exponent], characterized_gate.on(a, b), [cirq.Z(a) ** phase_exponent, cirq.Z(b) ** (-phase_exponent)]])))