import pytest
import datetime
import pytz
from traitlets import TraitError
from ..widget_datetime import NaiveDatetimePicker
def test_time_creation_value():
    t = datetime.datetime.today()
    w = NaiveDatetimePicker(value=t)
    assert w.value is t