import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_intrangeslider(page):
    widget = IntRangeSlider(start=1, end=10, step=1)
    serve_component(page, widget)
    page.locator('.noUi-touch-area').nth(0).click()
    for _ in range(3):
        page.keyboard.press('ArrowRight')
    wait_until(lambda: widget.value == (4, 10), page)