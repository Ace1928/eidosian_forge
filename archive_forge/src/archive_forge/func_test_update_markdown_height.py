from html import escape
import pytest
from playwright.sync_api import expect
from panel.models import HTML
from panel.pane import Markdown
from panel.tests.util import serve_component, wait_until
def test_update_markdown_height(page):
    md = Markdown('Initial', height=50)
    serve_component(page, md)
    md_el = page.locator('.markdown')
    expect(md_el.locator('div')).to_have_text('Initial\n')
    wait_until(lambda: md_el.bounding_box()['height'] == 50, page)
    md.height = 300
    wait_until(lambda: md_el.bounding_box()['height'] == 300, page)