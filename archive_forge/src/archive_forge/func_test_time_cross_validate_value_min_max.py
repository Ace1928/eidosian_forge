import pytest
import datetime
from traitlets import TraitError
from ..widget_time import TimePicker
def test_time_cross_validate_value_min_max():
    w = TimePicker(value=datetime.time(2), min=datetime.time(2), max=datetime.time(2))
    with w.hold_trait_notifications():
        w.value = None
        w.min = datetime.time(4)
        w.max = datetime.time(6)
    assert w.value is None
    with w.hold_trait_notifications():
        w.value = datetime.time(4)
        w.min = None
        w.max = None
    assert w.value == datetime.time(4)