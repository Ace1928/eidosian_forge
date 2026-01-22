import pytest
from playwright.sync_api import expect
from panel.pane import Markdown
from panel.template import EditableTemplate
from panel.tests.util import serve_component, wait_until
def test_editable_template_drag_item(page):
    tmpl = EditableTemplate()
    md1 = Markdown('1')
    md2 = Markdown('2')
    tmpl.main[:] = [md1, md2]
    serve_component(page, tmpl)
    md2_handle = page.locator('.muuri-handle.drag').nth(1)
    md2_handle.drag_to(md2_handle, target_position={'x': 0, 'y': -50}, force=True)
    wait_until(lambda: list(tmpl.layout) == [id(md2), id(md1)], page)