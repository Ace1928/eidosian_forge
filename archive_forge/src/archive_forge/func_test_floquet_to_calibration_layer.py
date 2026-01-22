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
def test_floquet_to_calibration_layer():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(gate=gate, pairs=((q_00, q_01), (q_02, q_03)), options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))
    assert request.to_calibration_layer() == cirq_google.CalibrationLayer(calibration_type='floquet_phased_fsim_characterization', program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)]), args={'est_theta': True, 'est_zeta': True, 'est_chi': False, 'est_gamma': False, 'est_phi': True, 'readout_corrections': True, 'version': 2})