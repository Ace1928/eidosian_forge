import numpy as np
from holoviews.element import BoxWhisker
from .test_plot import TestMPLPlot, mpl_renderer
def test_boxwhisker_simple_overlay(self):
    values = np.random.rand(100)
    boxwhisker = BoxWhisker(values) * BoxWhisker(values)
    plot = mpl_renderer.get_plot(boxwhisker)
    p1, p2 = plot.subplots.values()
    self.assertEqual(p1.handles['boxes'][0].get_path().vertices, p2.handles['boxes'][0].get_path().vertices)