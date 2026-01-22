import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def valid_prev_time(old_value, new_value, max_value, amount=1):
    current_value = int(old_value)
    prev_value = int(new_value)
    if current_value == 0:
        assert prev_value == max_value - amount
    else:
        assert prev_value == current_value - amount
    return True