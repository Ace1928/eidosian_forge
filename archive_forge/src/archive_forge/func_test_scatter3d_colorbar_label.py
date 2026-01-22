from holoviews.element import Scatter3D
from .test_plot import TestMPLPlot, mpl_renderer
def test_scatter3d_colorbar_label(self):
    scatter3d = Scatter3D([(0, 1, 2), (1, 2, 3), (2, 3, 4)]).opts(color='z', colorbar=True)
    plot = mpl_renderer.get_plot(scatter3d)
    assert plot.handles['cax'].get_ylabel() == 'z'