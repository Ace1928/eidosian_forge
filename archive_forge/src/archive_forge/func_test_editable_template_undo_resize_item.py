import pytest
from playwright.sync_api import expect
from panel.pane import Markdown
from panel.template import EditableTemplate
from panel.tests.util import serve_component, wait_until
def test_editable_template_undo_resize_item(page):
    md1 = Markdown('1')
    md2 = Markdown('2')
    tmpl = EditableTemplate(layout={id(md2): {'width': 50, 'height': 80}})
    tmpl.main[:] = [md1, md2]
    serve_component(page, tmpl)
    md2_handle = page.locator('.muuri-handle.resize').nth(1)
    md2_handle.hover()
    md2_handle.drag_to(md2_handle, target_position={'x': -50, 'y': -30}, force=True)
    wait_until(lambda: tmpl.layout.get(id(md2), {}).get('width') < 45, page)
    page.locator('#grid-undo').click()
    wait_until(lambda: tmpl.layout.get(id(md2), {}).get('width') == 50, page)