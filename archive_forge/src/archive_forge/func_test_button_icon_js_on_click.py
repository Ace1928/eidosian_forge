import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_button_icon_js_on_click(page):
    button = ButtonIcon()
    int_slider = IntInput(value=0)
    button.js_on_click(args={'int_slider': int_slider}, code='int_slider.value += 1')
    row = Row(button, int_slider)
    serve_component(page, row)
    page.click('.bk-TablerIcon')
    wait_until(lambda: int_slider.value == 1, page)
    assert int_slider.value == 1