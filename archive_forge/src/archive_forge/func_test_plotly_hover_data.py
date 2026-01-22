import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
def test_plotly_hover_data(page, plotly_2d_plot):
    hover_data = []
    plotly_2d_plot.param.watch(lambda e: hover_data.append(e.new), 'hover_data')
    serve_component(page, plotly_2d_plot)
    plotly_plot = page.locator('.js-plotly-plot .plot-container.plotly')
    expect(plotly_plot).to_have_count(1)
    point = plotly_plot.locator('g.points path.point').nth(0)
    point.hover(force=True)
    wait_until(lambda: {'points': [{'curveNumber': 0, 'pointIndex': 0, 'pointNumber': 0, 'x': 0, 'y': 2}]} in hover_data, page)
    plot = page.locator('.js-plotly-plot .plot-container.plotly g.scatterlayer')
    plot.hover(force=True)
    wait_until(lambda: plotly_2d_plot.hover_data is None, page)