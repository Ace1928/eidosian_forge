import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_scrollable(page):
    card = Card(scroll=True)
    serve_component(page, card)
    expect(page.locator('.card')).to_have_class('bk-panel-models-layout-Card card scrollable-vertical')