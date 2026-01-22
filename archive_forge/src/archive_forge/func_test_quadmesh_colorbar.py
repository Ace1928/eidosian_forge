import numpy as np
from bokeh.models import ColorBar
from holoviews.element import Image, QuadMesh
from .test_plot import TestBokehPlot, bokeh_renderer
def test_quadmesh_colorbar(self):
    n = 21
    xs = np.logspace(1, 3, n)
    ys = np.linspace(1, 10, n)
    qmesh = QuadMesh((xs, ys, np.random.rand(n - 1, n - 1))).opts(colorbar=True)
    plot = bokeh_renderer.get_plot(qmesh)
    self.assertIsInstance(plot.handles['colorbar'], ColorBar)
    self.assertIs(plot.handles['colorbar'].color_mapper, plot.handles['color_mapper'])