import param
import pytest
from playwright.sync_api import expect
from panel.reactive import ReactiveHTML
from panel.tests.util import serve_component, wait_until
def test_reactive_html_param_event(page):
    component = ReactiveComponent()
    serve_component(page, component)
    expect(page.locator('.reactive')).to_have_text('1')
    component.param.trigger('event')
    expect(page.locator('.reactive')).to_have_text('2')
    component.param.trigger('event')
    expect(page.locator('.reactive')).to_have_text('3')
    component.param.trigger('event')
    component.param.trigger('event')
    expect(page.locator('.reactive')).to_have_text('5')
    wait_until(lambda: component.count == 5, page)