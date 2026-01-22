import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
@pytest.fixture
def enabled_dates():
    enabled_list = [datetime.date(2021, 3, 1), datetime.date(2021, 3, 3)]
    enabled_str_list = ['March 1, 2021', 'March 3, 2021']
    active_date = datetime.datetime(2021, 3, 1)
    return (active_date, enabled_list, enabled_str_list)