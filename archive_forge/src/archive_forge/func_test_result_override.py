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
def test_result_override():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    result = PhasedFSimCalibrationResult(parameters={(q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=0.4, phi=0.5), (q_02, q_03): PhasedFSimCharacterization(theta=0.6, zeta=0.7, chi=None, gamma=0.9, phi=1.0)}, gate=gate, options=options)
    overridden = result.override(options.zeta_chi_gamma_correction_override())
    assert overridden == PhasedFSimCalibrationResult(parameters={(q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.0, chi=None, gamma=0.0, phi=0.5), (q_02, q_03): PhasedFSimCharacterization(theta=0.6, zeta=0.0, chi=None, gamma=0.0, phi=1.0)}, gate=gate, options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)