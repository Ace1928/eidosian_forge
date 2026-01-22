import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
@pytest.fixture
def months_as_str():
    months_str = '\n        January\n        February\n        March\n        April\n        May\n        June\n        July\n        August\n        September\n        October\n        November\n        December\n    '
    return months_str