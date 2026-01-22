import numpy as np
from bokeh.models import ColorBar
from holoviews.element import Image, QuadMesh
from .test_plot import TestBokehPlot, bokeh_renderer
def test_quadmesh_regular_centers(self):
    X = [0.5, 1.5]
    Y = [0.5, 1.5, 2.5]
    Z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]]).T
    LABELS = np.array([['0-0', '0-1', '0-2'], ['1-0', '1-1', '1-2']]).T
    qmesh = QuadMesh((X, Y, Z, LABELS), vdims=['Value', 'Label'])
    plot = bokeh_renderer.get_plot(qmesh.opts(tools=['hover']))
    source = plot.handles['source']
    expected = {'left': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 'right': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 'Value': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan], 'bottom': [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], 'top': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 'Label': ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2'], 'x': [0.5, 0.5, 0.5, 1.5, 1.5, 1.5], 'y': [0.5, 1.5, 2.5, 0.5, 1.5, 2.5]}
    self.assertEqual(source.data.keys(), expected.keys())
    for key in expected:
        self.assertEqual(list(source.data[key]), expected[key])