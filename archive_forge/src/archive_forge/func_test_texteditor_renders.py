import sys
import pytest
from playwright.sync_api import expect
from panel import depends
from panel.layout import Column
from panel.pane import HTML
from panel.tests.util import serve_component, wait_until
from panel.widgets import RadioButtonGroup, TextEditor
def test_texteditor_renders(page):
    widget = TextEditor()
    serve_component(page, widget)
    expect(page.locator('.ql-container')).to_be_visible()