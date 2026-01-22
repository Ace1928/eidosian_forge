import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
def test_plotly_img_plot(page, plotly_img_plot):
    msgs, _ = serve_component(page, plotly_img_plot)
    plotly_plot = page.locator('.js-plotly-plot .plot-container.plotly')
    expect(plotly_plot).to_have_count(1)
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []
    point = plotly_plot.locator('image')
    point.hover(force=True)
    wait_until(lambda: plotly_img_plot.hover_data == {'points': [{'curveNumber': 0, 'x': 15, 'y': 3, 'colormodel': 'rgb'}]}, page)