import sys
import pytest
from playwright.sync_api import expect
from panel import depends
from panel.layout import Column
from panel.pane import HTML
from panel.tests.util import serve_component, wait_until
from panel.widgets import RadioButtonGroup, TextEditor
def test_texteditor_regression_copy_paste(page):
    widget = TextEditor()
    html = HTML('test')
    serve_component(page, Column(html, widget))
    page.get_by_text('test').select_text()
    ctrl_key = 'Meta' if sys.platform == 'darwin' else 'Control'
    page.get_by_text('test').press(f'{ctrl_key}+KeyC')
    page.locator('.ql-editor').press(f'{ctrl_key}+KeyV')
    expect(page.locator('.ql-container')).to_have_text('test')
    wait_until(lambda: widget.value == '<p>test</p>', page)