import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_header_color_formatting(page):
    header_color = 'rgb(0, 0, 128)'
    active_header_background = 'rgb(0, 128, 0)'
    header_background = 'rgb(128, 0, 0)'
    card = Card(header_color=header_color, active_header_background=active_header_background, header_background=header_background)
    serve_component(page, card)
    card_header = page.locator('.card-header')
    expect(card_header).to_have_css('color', header_color)
    expect(card_header).to_have_css('background-color', active_header_background)
    card_header.wait_for()
    card_header.click()
    expect(card_header).to_have_css('background-color', header_background)