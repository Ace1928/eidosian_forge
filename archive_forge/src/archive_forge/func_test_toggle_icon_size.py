import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_toggle_icon_size(page):
    icon = ToggleIcon(size='120px')
    serve_component(page, icon)
    assert icon.icon == 'heart'
    assert not icon.value
    icon_element = page.locator('.ti-heart')
    wait_until(lambda: icon_element.bounding_box()['width'] == 120)