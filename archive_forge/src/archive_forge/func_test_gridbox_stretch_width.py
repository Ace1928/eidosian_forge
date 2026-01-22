import pytest
from panel import Column, GridBox, Spacer
from panel.tests.util import serve_component, wait_until
def test_gridbox_stretch_width(page):
    grid = Column(GridBox(Spacer(sizing_mode='stretch_width', height=50), Spacer(sizing_mode='stretch_width', height=50), Spacer(sizing_mode='stretch_width', height=50), ncols=2, sizing_mode='stretch_width'), width=800)
    serve_component(page, grid)
    bbox = page.locator('.bk-GridBox').bounding_box()
    children = page.locator('.bk-GridBox div')
    assert bbox['width'] == 800
    assert bbox['height'] == 100
    bbox1 = children.nth(0).bounding_box()
    assert bbox1['x'] == 0
    assert bbox1['width'] == 400
    assert bbox1['height'] == 50
    bbox2 = children.nth(1).bounding_box()
    assert bbox2['x'] == 400
    assert bbox2['width'] == 400
    assert bbox2['height'] == 50
    bbox3 = children.nth(2).bounding_box()
    assert bbox3['x'] == 0
    assert bbox3['width'] == 400
    assert bbox3['height'] == 50