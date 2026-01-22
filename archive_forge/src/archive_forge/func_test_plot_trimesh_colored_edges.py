import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
def test_plot_trimesh_colored_edges(self):
    opts = dict(edge_color_index='weight', edge_cmap='Greys')
    plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
    edges = plot.handles['edges']
    colors = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    self.assertEqual(edges.get_edgecolors(), colors)