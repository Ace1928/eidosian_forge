import pytest
from playwright.sync_api import expect
from panel.pane import Vega
from panel.tests.util import serve_component, wait_until
@altair_available
def test_altair_select_interval(page, dataframe):
    brush = alt.selection_interval(name='brush')
    chart = alt.Chart(dataframe).mark_circle().encode(x='int', y='float').add_params(brush)
    pane = Vega(chart)
    serve_component(page, pane)
    vega_plot = page.locator('.vega-embed')
    expect(vega_plot).to_have_count(1)
    bbox = vega_plot.bounding_box()
    vega_plot.click()
    page.mouse.move(bbox['x'] + 100, bbox['y'] + 100)
    page.mouse.down()
    page.mouse.move(bbox['x'] + 300, bbox['y'] + 300, steps=5)
    page.mouse.up()
    wait_until(lambda: pane.selection.brush == {'int': [0.61, 2.61], 'float': [0.33333333333333326, 7]}, page)