import logging
from io import StringIO
import pytest
from ..batteryrunners import BatteryRunner, Report
def test_init_report():
    rep = Report()
    assert rep == Report(Exception, 0, '', '')