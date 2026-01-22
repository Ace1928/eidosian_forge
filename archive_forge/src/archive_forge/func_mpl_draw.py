from collections.abc import Iterable
from itertools import islice, cycle
from numbers import Number
import numpy as np
import rustworkx
def mpl_draw(graph, pos=None, ax=None, arrows=True, with_labels=False, **kwds):
    """Draw a graph with Matplotlib.

    .. note::

        Matplotlib is an optional dependency and will not be installed with
        rustworkx by default. If you intend to use this function make sure that
        you install matplotlib with either ``pip install matplotlib`` or
        ``pip install 'rustworkx[mpl]'``

    :param graph: A rustworkx graph, either a :class:`~rustworkx.PyGraph` or a
        :class:`~rustworkx.PyDiGraph`.
    :param dict pos: An optional dictionary (or
        a :class:`~rustworkx.Pos2DMapping` object) with nodes as keys and
        positions as values. If not specified a spring layout positioning will
        be computed. See `layout_functions` for functions that compute
        node positions.
    :param matplotlib.Axes ax: An optional Matplotlib Axes object to draw the
        graph in.
    :param bool arrows: For :class:`~rustworkx.PyDiGraph` objects if ``True``
        draw arrowheads. (defaults to ``True``) Note, that the Arrows will
        be the same color as edges.
    :param str arrowstyle: An optional string for directed graphs to choose
        the style of the arrowsheads. See
        :class:`matplotlib.patches.ArrowStyle` for more options. By default the
        value is set to ``'-\\|>'``.
    :param int arrow_size: For directed graphs, choose the size of the arrow
        head's length and width. See
        :class:`matplotlib.patches.FancyArrowPatch` attribute and constructor
        kwarg ``mutation_scale`` for more info. Defaults to 10.
    :param bool with_labels: Set to ``True`` to draw labels on the nodes. Edge
        labels will only be drawn if the ``edge_labels`` parameter is set to a
        function. Defaults to ``False``.
    :param list node_list: An optional list of node indices in the graph to
        draw. If not specified all nodes will be drawn.
    :param list edge_list: An option list of edges in the graph to draw. If not
        specified all edges will be drawn
    :param int|list node_size: Optional size of nodes. If an array is
        specified it must be the same length as node_list. Defaults to 300
    :param node_color: Optional node color. Can be a single color or
        a sequence of colors with the same length as node_list. Color can be
        string or rgb (or rgba) tuple of floats from 0-1. If numeric values
        are specified they will be mapped to colors using the ``cmap`` and
        ``vmin``,``vmax`` parameters. See :func:`matplotlib.scatter` for more
        details. Defaults to ``'#1f78b4'``)
    :param str node_shape: The optional shape node. The specification is the
        same as the :func:`matplotlib.pyplot.scatter` function's ``marker``
        kwarg, valid options are one of
        ``['s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8']``. Defaults to
        ``'o'``
    :param float alpha: Optional value for node and edge transparency
    :param matplotlib.colors.Colormap cmap: An optional Matplotlib colormap
        object for mapping intensities of nodes
    :param float vmin: Optional minimum value for node colormap scaling
    :param float vmax: Optional minimum value for node colormap scaling
    :param float|sequence linewidths: An optional line width for symbol
        borders. If a sequence is specified it must be the same length as
        node_list. Defaults to 1.0
    :param float|sequence width: An optional width to use for edges. Can
        either be a float or sequence  of floats. If a sequence is specified
        it must be the same length as node_list. Defaults to 1.0
    :param str|sequence edge_color: color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edge_list. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the ``edge_cmap`` and ``edge_vmin``,
        ``edge_vmax`` parameters.
    :param matplotlib.colors.Colormap edge_cmap: An optional Matplotlib
        colormap for mapping intensities of edges.
    :param float edge_vmin: Optional minimum value for edge colormap scaling
    :param float edge_vmax: Optional maximum value for node colormap scaling
    :param str style: An optional string to specify the edge line style.
        For example, ``'-'``, ``'--'``, ``'-.'``, ``':'`` or words like
        ``'solid'`` or ``'dashed'``. See the
        :class:`matplotlib.patches.FancyArrowPatch` attribute and kwarg
        ``linestyle`` for more details. Defaults to ``'solid'``.
    :param func labels: An optional callback function that will be passed a
        node payload and return a string label for the node. For example::

            labels=str

        could be used to just return a string cast of the node's data payload.
        Or something like::

            labels=lambda node: node['label']

        could be used if the node payloads are dictionaries.
    :param func edge_labels: An optional callback function that will be passed
        an edge payload and return a string label for the edge. For example::

            edge_labels=str

        could be used to just return a string cast of the edge's data payload.
        Or something like::

            edge_labels=lambda edge: edge['label']

        could be used if the edge payloads are dictionaries. If this is set
        edge labels will be drawn in the visualization.
    :param int font_size: An optional fontsize to use for text labels, By
        default a value of 12 is used for nodes and 10 for edges.
    :param str font_color: An optional font color for strings. By default
        ``'k'`` (ie black) is set.
    :param str font_weight: An optional string used to specify the font weight.
        By default a value of ``'normal'`` is used.
    :param str font_family: An optional font family to use for strings. By
        default ``'sans-serif'`` is used.
    :param str label: An optional string label to use for the graph legend.
    :param str connectionstyle: An optional value used to create a curved arc
        of rounding radius rad. For example,
        ``connectionstyle='arc3,rad=0.2'``. See
        :class:`matplotlib.patches.ConnectionStyle` and
        :class:`matplotlib.patches.FancyArrowPatch` for more info. By default
        this is set to ``"arc3"``.

    :returns: A matplotlib figure for the visualization if not running with an
        interactive backend (like in jupyter) or if ``ax`` is not set.
    :rtype: matplotlib.figure.Figure

    For Example:

    .. jupyter-execute::

        import matplotlib.pyplot as plt

        import rustworkx as rx
        from rustworkx.visualization import mpl_draw

        G = rx.generators.directed_path_graph(25)
        mpl_draw(G)
        plt.draw()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib needs to be installed prior to running rustworkx.visualization.mpl_draw(). You can install matplotlib with:\n'pip install matplotlib'") from e
    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor('w')
    if ax is None:
        if cf.axes:
            ax = cf.gca()
        else:
            ax = cf.add_axes((0, 0, 1, 1))
    draw_graph(graph, pos=pos, ax=ax, arrows=arrows, with_labels=with_labels, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    if not plt.isinteractive() or ax is None:
        return cf