import pytest
from panel import Column, GridBox, Spacer
from panel.tests.util import serve_component, wait_until
def test_gridbox_stretch_both(page):
    grid = Column(GridBox(Spacer(sizing_mode='stretch_both'), Spacer(sizing_mode='stretch_both'), Spacer(sizing_mode='stretch_both'), ncols=2, sizing_mode='stretch_both'), height=800, width=600)
    serve_component(page, grid)
    bbox = page.locator('.bk-GridBox').bounding_box()
    children = page.locator('.bk-GridBox div')
    assert bbox['width'] == 600
    assert bbox['height'] == 800
    bbox1 = children.nth(0).bounding_box()
    assert bbox1['x'] == 0
    assert bbox1['y'] == 0
    assert bbox1['height'] == 400
    assert bbox1['width'] == 300
    bbox2 = children.nth(1).bounding_box()
    assert bbox2['x'] == 300
    assert bbox2['y'] == 0
    assert bbox2['height'] == 400
    assert bbox2['width'] == 300
    bbox3 = children.nth(2).bounding_box()
    assert bbox3['x'] == 0
    assert bbox3['y'] == 400
    assert bbox3['height'] == 400
    assert bbox3['width'] == 300