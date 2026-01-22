from html import escape
import pytest
from playwright.sync_api import expect
from panel.models import HTML
from panel.pane import Markdown
from panel.tests.util import serve_component, wait_until
def test_update_markdown_pane_resizes(page):
    md = Markdown('Initial')
    serve_component(page, md)
    height = page.locator('.markdown').bounding_box()['height']
    assert int(height) == 17
    md.object = '\n    - Bullet\n    - Points\n    '
    wait_until(lambda: int(page.locator('.markdown').bounding_box()['height']) == 34, page)