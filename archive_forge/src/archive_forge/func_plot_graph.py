from __future__ import division
import numpy as np
from pygsp import utils
def plot_graph(G, backend=None, **kwargs):
    """
    Plot a graph or a list of graphs.

    Parameters
    ----------
    G : Graph
        Graph to plot.
    show_edges : bool
        True to draw edges, false to only draw vertices.
        Default True if less than 10,000 edges to draw.
        Note that drawing a large number of edges might be particularly slow.
    backend: {'matplotlib', 'pyqtgraph'}
        Defines the drawing backend to use. Defaults to :data:`BACKEND`.
    vertex_size : float
        Size of circle representing each node.
    plot_name : str
        name of the plot
    save_as : str
        Whether to save the plot as save_as.png and save_as.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed. Only
        available with the matplotlib backend.

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Logo()
    >>> plotting.plot_graph(G)

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. Please run G.set_coordinates() first.')
    if G.coords.ndim != 2 or G.coords.shape[1] not in [2, 3]:
        raise AttributeError('Coordinates should be in 2D or 3D space.')
    kwargs['show_edges'] = kwargs.pop('show_edges', G.Ne < 10000.0)
    default = G.plotting['vertex_size']
    kwargs['vertex_size'] = kwargs.pop('vertex_size', default)
    plot_name = u'{}\nG.N={} nodes, G.Ne={} edges'.format(G.gtype, G.N, G.Ne)
    kwargs['plot_name'] = kwargs.pop('plot_name', plot_name)
    if backend is None:
        backend = BACKEND
    G = _handle_directed(G)
    if backend == 'pyqtgraph':
        _qtg_plot_graph(G, **kwargs)
    elif backend == 'matplotlib':
        _plt_plot_graph(G, **kwargs)
    else:
        raise ValueError('Unknown backend {}.'.format(backend))