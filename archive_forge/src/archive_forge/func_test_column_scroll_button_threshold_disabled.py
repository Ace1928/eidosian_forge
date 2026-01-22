import pytest
from playwright.sync_api import expect
from panel.layout.base import _SCROLL_MAPPING, Column
from panel.layout.spacer import Spacer
from panel.tests.util import serve_component, wait_until
def test_column_scroll_button_threshold_disabled(page):
    col = Column(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), scroll=True, scroll_button_threshold=0, height=420)
    serve_component(page, col)
    column = page.locator('.bk-panel-models-layout-Column')
    bbox = column.bounding_box()
    assert bbox['width'] in (200, 215)
    assert bbox['height'] == 420
    expect(column).to_have_class('bk-panel-models-layout-Column scrollable-vertical')
    scroll_arrow = page.locator('.scroll-button')
    expect(scroll_arrow).to_have_class('scroll-button')
    expect(scroll_arrow).not_to_be_visible()
    column.evaluate('(el) => el.scrollTo({top: 5})')
    expect(scroll_arrow).to_have_class('scroll-button')
    expect(scroll_arrow).not_to_be_visible()