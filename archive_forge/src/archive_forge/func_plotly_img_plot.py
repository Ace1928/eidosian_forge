import pytest
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from playwright.sync_api import expect
from panel.pane import Plotly
from panel.tests.util import serve_component, wait_until
@pytest.fixture
def plotly_img_plot():
    fig_dict = dict(data={'z': np.random.randint(0, 255, size=(6, 30, 3)).astype(np.uint8), 'type': 'image'}, layout={'width': 300, 'height': 60, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0}})
    return Plotly(fig_dict, width=300, height=60)