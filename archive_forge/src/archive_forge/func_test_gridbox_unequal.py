import pytest
from panel import Column, GridBox, Spacer
from panel.tests.util import serve_component, wait_until
def test_gridbox_unequal(page):
    grid = GridBox(Spacer(width=50, height=10), Spacer(width=40, height=10), Spacer(width=60, height=20), Spacer(width=20, height=10), Spacer(width=20, height=30), ncols=4)
    serve_component(page, grid)
    wait_until(lambda: page.locator('.bk-GridBox').bounding_box() == {'width': 170, 'height': 50, 'x': 0, 'y': 0}, page)
    children = page.locator('.bk-GridBox div')
    wait_until(lambda: children.nth(0).bounding_box() == {'x': 0, 'y': 0, 'width': 50, 'height': 10}, page)
    wait_until(lambda: children.nth(1).bounding_box() == {'x': 50, 'y': 0, 'width': 40, 'height': 10}, page)
    wait_until(lambda: children.nth(2).bounding_box() == {'x': 90, 'y': 0, 'width': 60, 'height': 20}, page)
    wait_until(lambda: children.nth(3).bounding_box() == {'x': 150, 'y': 0, 'width': 20, 'height': 10}, page)
    wait_until(lambda: children.nth(4).bounding_box() == {'x': 0, 'y': 20, 'width': 20, 'height': 30}, page)
    grid.ncols = 5
    wait_until(lambda: page.locator('.bk-GridBox').bounding_box() == {'width': 190, 'height': 30, 'x': 0, 'y': 0}, page)
    wait_until(lambda: children.nth(0).bounding_box() == {'x': 0, 'y': 0, 'width': 50, 'height': 10}, page)
    wait_until(lambda: children.nth(1).bounding_box() == {'x': 50, 'y': 0, 'width': 40, 'height': 10}, page)
    wait_until(lambda: children.nth(2).bounding_box() == {'x': 90, 'y': 0, 'width': 60, 'height': 20}, page)
    wait_until(lambda: children.nth(3).bounding_box() == {'x': 150, 'y': 0, 'width': 20, 'height': 10}, page)
    wait_until(lambda: children.nth(4).bounding_box() == {'x': 170, 'y': 0, 'width': 20, 'height': 30}, page)