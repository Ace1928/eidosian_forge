import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
def test_datetime_validate_max_vs_value():
    dt = dt_2002
    dt_min = dt_1664
    dt_max = dt_1994
    w = DatetimePicker(value=dt, min=dt_min)
    w.max = dt_max
    assert w.value.year == 1994