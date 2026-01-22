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
def test_plot_graph_categorical_colored_nodes(self):
    g = self.graph2.opts(color_index='Label', cmap='Set1')
    plot = mpl_renderer.get_plot(g)
    nodes = plot.handles['nodes']
    facecolors = np.array([[0.89411765, 0.10196078, 0.10980392, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0], [0.6, 0.6, 0.6, 1.0]])
    self.assertEqual(nodes.get_facecolors(), facecolors)