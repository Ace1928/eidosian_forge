import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
def test_plotly_select_data(page, plotly_2d_plot):
    serve_component(page, plotly_2d_plot)
    plotly_plot = page.locator('.js-plotly-plot .plot-container.plotly')
    expect(plotly_plot).to_have_count(1)
    page.locator('a.modebar-btn[data-val="select"]').click()
    bbox = page.locator('.js-plotly-plot .plot-container.plotly').bounding_box()
    page.mouse.move(bbox['x'] + 100, bbox['y'] + 100)
    page.mouse.down()
    page.mouse.move(bbox['x'] + bbox['width'], bbox['y'] + bbox['height'], steps=5)
    page.mouse.up()
    wait_until(lambda: plotly_2d_plot.selected_data is not None, page)
    selected = plotly_2d_plot.selected_data
    assert selected is not None
    assert 'points' in selected
    assert selected['points'] == [{'curveNumber': 0, 'pointIndex': 1, 'pointNumber': 1, 'x': 1, 'y': 3}]
    assert 'range' in selected
    assert 'x' in selected['range']
    assert 'y' in selected['range']