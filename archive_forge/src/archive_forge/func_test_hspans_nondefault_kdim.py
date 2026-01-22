import numpy as np
import holoviews as hv
from holoviews.element import HLines, HSpans, VLines, VSpans
from .test_plot import TestMPLPlot, mpl_renderer
def test_hspans_nondefault_kdim(self):
    hspans = HSpans({'other0': [0, 3, 5.5], 'other1': [1, 4, 6.5]}, kdims=['other0', 'other1'])
    plot = mpl_renderer.get_plot(hspans)
    assert plot.handles['fig'].axes[0].get_xlabel() == 'x'
    assert plot.handles['fig'].axes[0].get_ylabel() == 'y'
    xlim = plot.handles['fig'].axes[0].get_xlim()
    ylim = plot.handles['fig'].axes[0].get_ylim()
    assert np.allclose(xlim, (-0.055, 0.055))
    assert np.allclose(ylim, (0, 6.5))
    sources = plot.handles['annotations']
    assert len(sources) == 3
    for source, v0, v1 in zip(sources, hspans.data['other0'], hspans.data['other1']):
        assert np.allclose(source.xy[:, 0], [0, 0, 1, 1, 0])
        assert np.allclose(source.xy[:, 1], [v0, v1, v1, v0, v0])