import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
@pytest.mark.usefixtures('closefigures')
def test_calibration_plot():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    _, axs = calibration.plot('two_qubit_xeb')
    assert axs[0].get_title() == 'Two Qubit Xeb'
    assert len(axs[1].get_lines()) == 2