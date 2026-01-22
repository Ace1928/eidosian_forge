import numpy as np
import holoviews as hv
from holoviews.element import HLines, HSpans, VLines, VSpans
from .test_plot import TestMPLPlot, mpl_renderer
def test_hspans_inverse_plot(self):
    hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra']).opts(invert_axes=True)
    plot = mpl_renderer.get_plot(hspans)
    assert plot.handles['fig'].axes[0].get_xlabel() == 'y'
    assert plot.handles['fig'].axes[0].get_ylabel() == 'x'
    xlim = plot.handles['fig'].axes[0].get_xlim()
    ylim = plot.handles['fig'].axes[0].get_ylim()
    assert np.allclose(xlim, (0, 6.5))
    assert np.allclose(ylim, (-0.055, 0.055))
    sources = plot.handles['annotations']
    assert len(sources) == 3
    for source, v0, v1 in zip(sources, hspans.data['y0'], hspans.data['y1']):
        assert np.allclose(source.xy[:, 1], [0, 1, 1, 0, 0])
        assert np.allclose(source.xy[:, 0], [v0, v0, v1, v1, v0])