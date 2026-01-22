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
def test_to_zphase_unknown_gate_raises_error():
    q0, q1, q2 = cirq.GridQubit.rect(1, 3)
    result_1 = PhasedFSimCalibrationResult({(q0, q1): PhasedFSimCharacterization(zeta=0.1, gamma=0.2), (q1, q2): PhasedFSimCharacterization(zeta=0.3, gamma=0.4)}, gate=cirq.CZPowGate(), options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)
    with pytest.raises(ValueError, match="Only 'SycamoreGate' and 'ISwapPowGate' are supported"):
        _ = to_zphase_data([result_1])