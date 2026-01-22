import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_calibrations_with_string_key():
    calibration = cg.Calibration(metrics={'metric1': {('alpha',): [0.1]}})
    expected_proto = Merge("\n        metrics: [{\n          name: 'metric1'\n          targets: ['alpha']\n          values: [{double_val: 0.1}]\n        }]\n    ", v2.metrics_pb2.MetricsSnapshot())
    assert expected_proto == calibration.to_proto()
    assert calibration == cg.Calibration(expected_proto)
    assert calibration == cg.Calibration(calibration.to_proto())