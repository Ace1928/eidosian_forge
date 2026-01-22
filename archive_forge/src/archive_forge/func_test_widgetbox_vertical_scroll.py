import pytest
from panel import Spacer, WidgetBox
from panel.tests.util import serve_component
def test_widgetbox_vertical_scroll(page):
    wbox = WidgetBox(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), scroll=True, height=420)
    serve_component(page, wbox)
    bbox = page.locator('.bk-panel-models-layout-Column').bounding_box()
    assert bbox['width'] in (202, 217)
    assert bbox['height'] == 420