import pytest
import datetime
import pytz
from traitlets import TraitError
from ..widget_datetime import NaiveDatetimePicker
def test_time_validate_value_none():
    t = datetime.datetime(2002, 2, 20, 13, 37, 42, 7)
    t_min = datetime.datetime(1442, 1, 1)
    t_max = datetime.datetime(2056, 1, 1)
    w = NaiveDatetimePicker(value=t, min=t_min, max=t_max)
    w.value = None
    assert w.value is None