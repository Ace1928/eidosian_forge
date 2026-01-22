import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def test_accordion_append(page, accordion_components):
    accordion = Accordion()
    serve_component(page, accordion)
    accordion_elements = page.locator('.accordion')
    expect(accordion_elements).to_have_count(0)
    d0, d1 = accordion_components
    accordion.append(d0)
    expect(accordion_elements).to_have_count(1)
    d0_object = accordion_elements.nth(0)
    assert is_collapsed(card_object=d0_object, card_content=d0.text)
    accordion.append(d1)
    expect(accordion_elements).to_have_count(2)
    d1_object = accordion_elements.nth(1)
    assert is_collapsed(card_object=d0_object, card_content=d0.text)
    assert is_collapsed(card_object=d1_object, card_content=d1.text)