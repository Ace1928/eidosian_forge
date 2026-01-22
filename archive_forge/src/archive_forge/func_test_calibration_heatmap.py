import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_calibration_heatmap():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    heatmap = calibration.heatmap('t1')
    figure = mpl.figure.Figure()
    axes = figure.add_subplot(111)
    heatmap.plot(axes)
    assert axes.get_title() == 'T1'
    heatmap = calibration.heatmap('two_qubit_xeb')
    figure = mpl.figure.Figure()
    axes = figure.add_subplot(999)
    heatmap.plot(axes)
    assert axes.get_title() == 'Two Qubit Xeb'
    with pytest.raises(ValueError, match='one or two qubits.*multi_qubit'):
        multi_qubit_data = Merge("metrics: [{\n                name: 'multi_qubit',\n                targets: ['0_0', '0_1', '1_0'],\n                values: [{double_val: 0.999}]}]", v2.metrics_pb2.MetricsSnapshot())
        cg.Calibration(multi_qubit_data).heatmap('multi_qubit')
    with pytest.raises(ValueError, match='single metric values.*multi_value'):
        multi_qubit_data = Merge("metrics: [{\n                name: 'multi_value',\n                targets: ['0_0'],\n                values: [{double_val: 0.999}, {double_val: 0.001}]}]", v2.metrics_pb2.MetricsSnapshot())
        cg.Calibration(multi_qubit_data).heatmap('multi_value')