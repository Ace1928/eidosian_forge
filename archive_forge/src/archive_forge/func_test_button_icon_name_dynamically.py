import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_button_icon_name_dynamically(page):
    button = ButtonIcon(name='Like')
    serve_component(page, button)
    assert button.name == 'Like'
    assert page.locator('.bk-IconLabel').text_content() == 'Like'
    button.name = 'Dislike'
    assert button.name == 'Dislike'
    assert page.locator('.bk-IconLabel').text_content() == 'Dislike'
    assert page.locator('.bk-IconLabel').bounding_box()['width'] < 40
    button.size = '2em'
    assert page.locator('.bk-IconLabel').bounding_box()['width'] >= 40