import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
@pytest.mark.usefixtures('closefigures')
def test_calibration_plot_histograms():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    _, ax = plt.subplots(1, 1)
    calibration.plot_histograms(['t1', 'two_qubit_xeb'], ax, labels=['T1', 'XEB'])
    assert len(ax.get_lines()) == 4
    with pytest.raises(ValueError, match='single metric values.*multi_value'):
        multi_qubit_data = Merge("metrics: [{\n                name: 'multi_value',\n                targets: ['0_0'],\n                values: [{double_val: 0.999}, {double_val: 0.001}]}]", v2.metrics_pb2.MetricsSnapshot())
        cg.Calibration(multi_qubit_data).plot_histograms('multi_value')