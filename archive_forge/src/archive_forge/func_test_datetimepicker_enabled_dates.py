import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_enabled_dates(page, march_2021, enabled_dates):
    active_date, enabled_list, enabled_str_list = enabled_dates
    datetime_picker_widget = DatetimePicker(enabled_dates=enabled_list, value=active_date)
    serve_component(page, datetime_picker_widget)
    datetime_value = page.locator('.flatpickr-input')
    datetime_value.dblclick()
    _, num_days, _, _ = march_2021
    disabled_days = page.locator('.flatpickr-calendar .flatpickr-day.flatpickr-disabled')
    expect(disabled_days).to_have_count(num_days - len(enabled_list))
    datetime_picker_widget.enabled_dates = None
    disabled_days = page.locator('.flatpickr-calendar .flatpickr-day.flatpickr-disabled')
    expect(disabled_days).to_have_count(0)