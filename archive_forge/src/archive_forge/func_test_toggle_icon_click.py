import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_toggle_icon_click(page):
    icon = ToggleIcon()
    serve_component(page, icon)
    assert icon.icon == 'heart'
    assert not icon.value
    icon_element = page.locator('.ti-heart')
    assert icon_element
    events = []

    def cb(event):
        events.append(event)
    icon.param.watch(cb, 'value')
    page.click('.bk-TablerIcon')
    wait_until(lambda: len(events) == 1, page)
    assert icon.value