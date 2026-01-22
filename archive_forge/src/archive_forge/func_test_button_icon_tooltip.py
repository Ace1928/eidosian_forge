import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_button_icon_tooltip(page):
    button = ButtonIcon(description='Click me')
    serve_component(page, button)
    page.hover('.bk-TablerIcon')
    wait_until(lambda: page.locator('.bk-tooltip-content') is not None, page)
    assert page.locator('.bk-tooltip-content').text_content() == 'Click me'