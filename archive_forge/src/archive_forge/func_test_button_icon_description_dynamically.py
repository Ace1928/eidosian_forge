import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_button_icon_description_dynamically(page):
    button = ButtonIcon(description='Like')
    serve_component(page, button)
    assert button.description == 'Like'
    page.locator('.bk-TablerIcon').click()
    assert page.locator('.bk-tooltip-content').text_content() == 'Like'
    button.description = 'Dislike'
    page.locator('.bk-TablerIcon').click()
    assert page.locator('.bk-tooltip-content').text_content() == 'Dislike'