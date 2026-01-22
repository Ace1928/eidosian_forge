import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
@pytest.fixture
def march_2021():
    march_2021_days = [28, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    march_2021_str = '\n'.join([str(i) for i in march_2021_days])
    num_days = 42
    num_prev_month_days = 1
    num_next_month_days = 10
    return (march_2021_str, num_days, num_prev_month_days, num_next_month_days)