import param
import pytest
from playwright.sync_api import expect
from panel.reactive import ReactiveHTML
from panel.tests.util import serve_component, wait_until
def test_reactive_html_changing_css_classes_rerenders(page):
    component = ReactiveComponent()
    serve_component(page, component)
    expect(page.locator('.reactive')).to_have_text('1')
    component.css_classes = ['custom']
    expect(page.locator('.reactive')).to_have_text('1')
    component.loading = True
    expect(page.locator('.reactive')).to_have_text('1')
    component.css_classes = []
    expect(page.locator('.reactive')).to_have_text('1')