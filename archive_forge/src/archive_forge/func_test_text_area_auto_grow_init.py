import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_text_area_auto_grow_init(page):
    text_area = TextAreaInput(auto_grow=True, value='1\n2\n3\n4\n')
    serve_component(page, text_area)
    expect(page.locator('.bk-input')).to_have_js_property('rows', 5)