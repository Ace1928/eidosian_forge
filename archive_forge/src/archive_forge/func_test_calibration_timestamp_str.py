import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_calibration_timestamp_str():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    assert calibration.timestamp_str(tz=datetime.timezone.utc) == '2019-07-08 00:00:00.021021+00:00'
    assert calibration.timestamp_str(tz=datetime.timezone(datetime.timedelta(hours=1))) == '2019-07-08 01:00:00.021021+01:00'