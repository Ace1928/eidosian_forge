import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def is_collapsed(card_object, card_content):
    expect(card_object.locator('svg')).to_have_class('icon icon-tabler icons-tabler-outline icon-tabler-chevron-right')
    expect(card_object).not_to_contain_text(card_content)
    return True