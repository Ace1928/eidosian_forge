import pytest
from playwright.sync_api import expect
from panel.layout.base import _SCROLL_MAPPING, Column
from panel.layout.spacer import Spacer
from panel.tests.util import serve_component, wait_until
def test_column_auto_scroll_limit(page):
    col = Column(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), auto_scroll_limit=100, height=420)
    serve_component(page, col)
    column = page.locator('.bk-panel-models-layout-Column')
    bbox = column.bounding_box()
    assert bbox['width'] in (200, 215)
    assert bbox['height'] == 420
    expect(column.locator('div')).to_have_count(4)
    expect(column).to_have_class('bk-panel-models-layout-Column scrollable-vertical')
    expect(column).to_have_js_property('scrollTop', 0)
    col.append(Spacer(styles=dict(background='yellow'), width=200, height=200))
    page.wait_for_timeout(500)
    expect(column.locator('div')).to_have_count(5)
    expect(column).to_have_js_property('scrollTop', 0)
    column.evaluate('(el) => el.scrollTo({top: el.scrollHeight})')
    col.append(Spacer(styles=dict(background='yellow'), width=200, height=200))
    wait_until(lambda: column.evaluate('(el) => el.scrollHeight - el.scrollTop - el.clientHeight') == 0, page)