import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_text_area_auto_grow_shrink_back_on_new_value(page):
    text_area = TextAreaInput(auto_grow=True, value='1\n2\n3\n4\n', max_rows=5)
    serve_component(page, text_area)
    input_area = page.locator('.bk-input')
    input_area.click()
    for _ in range(5):
        input_area.press('ArrowDown')
    for _ in range(10):
        input_area.press('Backspace')
    text_area.value = ''
    expect(page.locator('.bk-input')).to_have_js_property('rows', 2)