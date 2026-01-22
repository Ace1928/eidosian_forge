import numpy as np
import PIL.Image
import plotly.graph_objs as go
from holoviews.element import RGB, Tiles
from .test_plot import TestPlotlyPlot, plotly_renderer
@staticmethod
def rgb_element_to_pil_img(rgb_data):
    return PIL.Image.fromarray(np.clip(rgb_data * 255, 0, 255).astype('uint8'))