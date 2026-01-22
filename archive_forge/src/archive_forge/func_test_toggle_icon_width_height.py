import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_toggle_icon_width_height(page):
    icon = ToggleIcon(width=100, height=100)
    serve_component(page, icon)
    assert icon.icon == 'heart'
    assert not icon.value
    icon_element = page.locator('.ti-heart')
    wait_until(lambda: icon_element.bounding_box()['width'] == 100)