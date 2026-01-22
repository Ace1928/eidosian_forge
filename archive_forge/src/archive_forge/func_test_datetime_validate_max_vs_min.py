import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
def test_datetime_validate_max_vs_min():
    dt = dt_2002
    dt_min = dt_1664
    dt_max = datetime.datetime(1337, 1, 1, tzinfo=pytz.utc)
    w = DatetimePicker(value=dt, min=dt_min)
    with pytest.raises(TraitError):
        w.max = dt_max