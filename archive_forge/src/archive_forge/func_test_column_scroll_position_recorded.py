import pytest
from playwright.sync_api import expect
from panel.layout.base import _SCROLL_MAPPING, Column
from panel.layout.spacer import Spacer
from panel.tests.util import serve_component, wait_until
def test_column_scroll_position_recorded(page):
    col = Column(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), scroll=True, height=420)
    serve_component(page, col)
    column = page.locator('.bk-panel-models-layout-Column')
    column.evaluate('(el) => el.scrollTop = 150')
    expect(column).to_have_js_property('scrollTop', 150)