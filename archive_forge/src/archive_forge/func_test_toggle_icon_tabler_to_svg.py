import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_toggle_icon_tabler_to_svg(page):
    tabler = 'ad-off'
    icon = ToggleIcon(icon=tabler, active_icon=ACTIVE_SVG)
    serve_component(page, icon)
    assert icon.icon == tabler
    assert not icon.value
    assert page.locator('.icon-tabler-ad-off')
    events = []

    def cb(event):
        events.append(event)
    icon.param.watch(cb, 'value')
    page.click('.bk-TablerIcon')
    wait_until(lambda: len(events) == 1, page)
    assert icon.value
    assert page.locator('.icon-tabler-ad-filled')