import pytest
from panel import Spacer, WidgetBox
from panel.tests.util import serve_component
def test_widgetbox_horizontal_scroll(page):
    wbox = WidgetBox(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), scroll=True, width=420, horizontal=True)
    serve_component(page, wbox)
    bbox = page.locator('.bk-Row').bounding_box()
    assert bbox['width'] == 420
    assert bbox['height'] in (202, 217)