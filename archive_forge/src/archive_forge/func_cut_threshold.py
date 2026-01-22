import networkx as nx
import numpy as np
from scipy.sparse import linalg
from . import _ncut, _ncut_cy
def cut_threshold(labels, rag, thresh, in_place=True):
    """Combine regions separated by weight less than threshold.

    Given an image's labels and its RAG, output new labels by
    combining regions whose nodes are separated by a weight less
    than the given threshold.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. Regions connected by edges with smaller weights are
        combined.
    in_place : bool
        If set, modifies `rag` in place. The function will remove the edges
        with weights less that `thresh`. If set to `False` the function
        makes a copy of `rag` before proceeding.

    Returns
    -------
    out : ndarray
        The new labelled array.

    Examples
    --------
    >>> from skimage import data, segmentation, graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    >>> new_labels = graph.cut_threshold(labels, rag, 10)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           :DOI:`10.1109/83.841950`

    """
    if not in_place:
        rag = rag.copy()
    to_remove = [(x, y) for x, y, d in rag.edges(data=True) if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)
    comps = nx.connected_components(rag)
    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.nodes[node]['labels']:
                map_array[label] = i
    return map_array[labels]