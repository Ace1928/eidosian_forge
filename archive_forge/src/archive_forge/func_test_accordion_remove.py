import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def test_accordion_remove(page, accordion_components):
    d0, d1 = accordion_components
    accordion = Accordion(d0, d1)
    serve_component(page, accordion)
    accordion_elements = page.locator('.accordion')
    expect(accordion_elements).to_have_count(len(accordion_components))
    accordion.remove(pane=accordion.objects[0])
    expect(accordion_elements).to_have_count(len(accordion_components) - 1)
    d1_object = accordion_elements.nth(0)
    expect(d1_object).to_contain_text(d1.name)