import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_enable_time(page):
    datetime_picker_widget = DatetimePicker(enable_time=False)
    serve_component(page, datetime_picker_widget)
    page.locator('.flatpickr-input').dblclick()
    time_editor = page.locator('.flatpickr-calendar .flatpickr-time.time24hr.hasSeconds')
    expect(time_editor).to_have_count(0)