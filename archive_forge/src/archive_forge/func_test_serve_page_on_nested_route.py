import pytest
from panel.config import panel_extension as extension
from panel.pane import Markdown
from panel.tests.util import serve_component
def test_serve_page_on_nested_route(page):
    md = Markdown('Initial')
    msgs, _ = serve_component(page, {'/foo/bar': md})
    expect(page.locator('.markdown').locator('div')).to_have_text('Initial\n')
    md.object = 'Updated'
    expect(page.locator('.markdown').locator('div')).to_have_text('Updated\n')
    assert [msg for msg in msgs if msg.type == 'error'] == []