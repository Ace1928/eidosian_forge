import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def test_accordion_objects(page, accordion_components):
    d0, d1 = accordion_components
    accordion = Accordion(d0, d1)
    serve_component(page, accordion)
    new_objects = [d0]
    accordion.objects = new_objects
    accordion_elements = page.locator('.accordion')
    expect(accordion_elements).to_have_count(len(new_objects))