import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_text_area_auto_grow_max_rows(page):
    text_area = TextAreaInput(auto_grow=True, value='1\n2\n3\n4\n', max_rows=7)
    serve_component(page, text_area)
    input_area = page.locator('.bk-input')
    input_area.click()
    input_area.press('Enter')
    input_area.press('Enter')
    input_area.press('Enter')
    expect(page.locator('.bk-input')).to_have_js_property('rows', 7)