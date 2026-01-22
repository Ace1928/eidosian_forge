import pytest
import datetime
import pytz
from traitlets import TraitError
from ..widget_datetime import NaiveDatetimePicker
def test_time_validate_max_vs_min():
    t = datetime.datetime(2002, 2, 20, 13, 37, 42, 7)
    t_min = datetime.datetime(1664, 1, 1)
    t_max = datetime.datetime(1337, 1, 1)
    w = NaiveDatetimePicker(value=t, min=t_min)
    with pytest.raises(TraitError):
        w.max = t_max