import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from ..morphology._util import _raveled_offsets_and_distances
from ..util._map_array import map_array
def pixel_graph(image, *, mask=None, edge_function=None, connectivity=1, spacing=None):
    """Create an adjacency graph of pixels in an image.

    Pixels where the mask is True are nodes in the returned graph, and they are
    connected by edges to their neighbors according to the connectivity
    parameter. By default, the *value* of an edge when a mask is given, or when
    the image is itself the mask, is the Euclidean distance between the pixels.

    However, if an int- or float-valued image is given with no mask, the value
    of the edges is the absolute difference in intensity between adjacent
    pixels, weighted by the Euclidean distance.

    Parameters
    ----------
    image : array
        The input image. If the image is of type bool, it will be used as the
        mask as well.
    mask : array of bool
        Which pixels to use. If None, the graph for the whole image is used.
    edge_function : callable
        A function taking an array of pixel values, and an array of neighbor
        pixel values, and an array of distances, and returning a value for the
        edge. If no function is given, the value of an edge is just the
        distance.
    connectivity : int
        The square connectivity of the pixel neighborhood: the number of
        orthogonal steps allowed to consider a pixel a neighbor. See
        `scipy.ndimage.generate_binary_structure` for details.
    spacing : tuple of float
        The spacing between pixels along each axis.

    Returns
    -------
    graph : scipy.sparse.csr_matrix
        A sparse adjacency matrix in which entry (i, j) is 1 if nodes i and j
        are neighbors, 0 otherwise.
    nodes : array of int
        The nodes of the graph. These correspond to the raveled indices of the
        nonzero pixels in the mask.
    """
    if mask is None:
        if image.dtype == bool:
            mask = image
        else:
            mask = np.ones_like(image, dtype=bool)
    if edge_function is None:
        if image.dtype == bool:

            def edge_function(x, y, distances):
                return distances
        else:
            edge_function = _weighted_abs_diff
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    nodes_padded = np.flatnonzero(padded)
    neighbor_offsets_padded, distances_padded = _raveled_offsets_and_distances(padded.shape, connectivity=connectivity, spacing=spacing)
    neighbors_padded = nodes_padded[:, np.newaxis] + neighbor_offsets_padded
    neighbor_distances_full = np.broadcast_to(distances_padded, neighbors_padded.shape)
    nodes = np.flatnonzero(mask)
    nodes_sequential = np.arange(nodes.size)
    neighbors = map_array(neighbors_padded, nodes_padded, nodes)
    neighbors_mask = padded.reshape(-1)[neighbors_padded]
    num_neighbors = np.sum(neighbors_mask, axis=1)
    indices = np.repeat(nodes, num_neighbors)
    indices_sequential = np.repeat(nodes_sequential, num_neighbors)
    neighbor_indices = neighbors[neighbors_mask]
    neighbor_distances = neighbor_distances_full[neighbors_mask]
    neighbor_indices_sequential = map_array(neighbor_indices, nodes, nodes_sequential)
    image_r = image.reshape(-1)
    data = edge_function(image_r[indices], image_r[neighbor_indices], neighbor_distances)
    m = nodes_sequential.size
    mat = sparse.coo_matrix((data, (indices_sequential, neighbor_indices_sequential)), shape=(m, m))
    graph = mat.tocsr()
    return (graph, nodes)