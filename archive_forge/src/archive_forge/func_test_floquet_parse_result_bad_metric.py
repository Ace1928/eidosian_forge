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
def test_floquet_parse_result_bad_metric():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(gate=gate, pairs=((q_00, q_01), (q_02, q_03)), options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))
    result = cirq_google.CalibrationResult(code=cirq_google.api.v2.calibration_pb2.SUCCESS, error_message=None, token=None, valid_until=None, metrics=cirq_google.Calibration(cirq_google.api.v2.metrics_pb2.MetricsSnapshot(metrics=[cirq_google.api.v2.metrics_pb2.Metric(name='angles', targets=['1000gerbils'], values=[cirq_google.api.v2.metrics_pb2.Value(str_val='100_10')])])))
    with pytest.raises(ValueError, match='Unknown metric name 1000gerbils'):
        _ = request.parse_result(result)