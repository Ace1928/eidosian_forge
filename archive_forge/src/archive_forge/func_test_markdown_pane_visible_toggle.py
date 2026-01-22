from html import escape
import pytest
from playwright.sync_api import expect
from panel.models import HTML
from panel.pane import Markdown
from panel.tests.util import serve_component, wait_until
def test_markdown_pane_visible_toggle(page):
    md = Markdown('Initial', visible=False)
    serve_component(page, md)
    assert page.locator('.markdown').locator('div').text_content() == 'Initial\n'
    assert not page.locator('.markdown').locator('div').is_visible()
    md.visible = True
    wait_until(lambda: page.locator('.markdown').locator('div').is_visible(), page)