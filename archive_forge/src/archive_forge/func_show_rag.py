import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
@require('matplotlib', '>=3.3')
def show_rag(labels, rag, image, border_color='black', edge_width=1.5, edge_cmap='magma', img_cmap='bone', in_place=True, ax=None):
    """Show a Region Adjacency Graph on an image.

    Given a labelled image and its corresponding RAG, show the nodes and edges
    of the RAG on the image with the specified colors. Edges are displayed between
    the centroid of the 2 adjacent regions in the image.

    Parameters
    ----------
    labels : ndarray, shape (M, N)
        The labelled image.
    rag : RAG
        The Region Adjacency Graph.
    image : ndarray, shape (M, N[, 3])
        Input image. If `colormap` is `None`, the image should be in RGB
        format.
    border_color : color spec, optional
        Color with which the borders between regions are drawn.
    edge_width : float, optional
        The thickness with which the RAG edges are drawn.
    edge_cmap : :py:class:`matplotlib.colors.Colormap`, optional
        Any matplotlib colormap with which the edges are drawn.
    img_cmap : :py:class:`matplotlib.colors.Colormap`, optional
        Any matplotlib colormap with which the image is draw. If set to `None`
        the image is drawn as it is.
    in_place : bool, optional
        If set, the RAG is modified in place. For each node `n` the function
        will set a new attribute ``rag.nodes[n]['centroid']``.
    ax : :py:class:`matplotlib.axes.Axes`, optional
        The axes to draw on. If not specified, new axes are created and drawn
        on.

    Returns
    -------
    lc : :py:class:`matplotlib.collections.LineCollection`
         A collection of lines that represent the edges of the graph. It can be
         passed to the :meth:`matplotlib.figure.Figure.colorbar` function.

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('matplotlib')

    >>> from skimage import data, segmentation, graph
    >>> import matplotlib.pyplot as plt
    >>>
    >>> img = data.coffee()
    >>> labels = segmentation.slic(img)
    >>> g =  graph.rag_mean_color(img, labels)
    >>> lc = graph.show_rag(labels, g, img)
    >>> cbar = plt.colorbar(lc)
    """
    from matplotlib import colors
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    if not in_place:
        rag = rag.copy()
    if ax is None:
        fig, ax = plt.subplots()
    out = util.img_as_float(image, force_copy=True)
    if img_cmap is None:
        if image.ndim < 3 or image.shape[2] not in [3, 4]:
            msg = 'If colormap is `None`, an RGB or RGBA image should be given'
            raise ValueError(msg)
        out = image[:, :, :3]
    else:
        img_cmap = plt.get_cmap(img_cmap)
        out = color.rgb2gray(image)
        out = img_cmap(out)[:, :, :3]
    edge_cmap = plt.get_cmap(edge_cmap)
    offset = 1
    map_array = np.arange(labels.max() + 1)
    for n, d in rag.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1
    rag_labels = map_array[labels]
    regions = measure.regionprops(rag_labels)
    for (n, data), region in zip(rag.nodes(data=True), regions):
        data['centroid'] = tuple(map(int, region['centroid']))
    cc = colors.ColorConverter()
    if border_color is not None:
        border_color = cc.to_rgb(border_color)
        out = segmentation.mark_boundaries(out, rag_labels, color=border_color)
    ax.imshow(out)
    lines = [[rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]] for n1, n2 in rag.edges()]
    lc = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
    edge_weights = [d['weight'] for x, y, d in rag.edges(data=True)]
    lc.set_array(np.array(edge_weights))
    ax.add_collection(lc)
    return lc