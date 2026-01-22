import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def tree_multiresolution(G, Nlevel, reduction_method='resistance_distance', compute_full_eigen=False, root=None):
    """Compute a multiresolution of trees

    Parameters
    ----------
    G : Graph
        Graph structure of a tree.
    Nlevel : Number of times to downsample and coarsen the tree
    root : int
        The index of the root of the tree. (default = 1)
    reduction_method : str
        The graph reduction method (default = 'resistance_distance')
    compute_full_eigen : bool
        To also compute the graph Laplacian eigenvalues for every tree in the sequence

    Returns
    -------
    Gs : ndarray
        Ndarray, with each element containing a graph structure represent a reduced tree.
    subsampled_vertex_indices : ndarray
        Indices of the vertices of the previous tree that are kept for the subsequent tree.

    """
    if not root:
        if hasattr(G, 'root'):
            root = G.root
        else:
            root = 1
    Gs = [G]
    if compute_full_eigen:
        Gs[0].compute_fourier_basis()
    subsampled_vertex_indices = []
    depths, parents = _tree_depths(G.A, root)
    old_W = G.W
    for lev in range(Nlevel):
        down_odd = round(depths) % 2
        down_even = np.ones(Gs[lev].N) - down_odd
        keep_inds = np.where(down_even == 1)[0]
        subsampled_vertex_indices.append(keep_inds)
        non_root_keep_inds, new_non_root_inds = np.setdiff1d(keep_inds, root)
        old_parents_of_non_root_keep_inds = parents[non_root_keep_inds]
        old_grandparents_of_non_root_keep_inds = parents[old_parents_of_non_root_keep_inds]
        old_W_i_inds, old_W_j_inds, old_W_weights = sparse.find(old_W)
        i_inds = np.concatenate((new_non_root_inds, new_non_root_parents))
        j_inds = np.concatenate((new_non_root_parents, new_non_root_inds))
        new_N = np.sum(down_even)
        if reduction_method == 'unweighted':
            new_weights = np.ones(np.shape(i_inds))
        elif reduction_method == 'sum':
            old_weights_to_parents = old_W_weights[old_weights_to_parents_inds]
            old_weights_parents_to_grandparents = old_W_weights[old_weights_parents_to_grandparents_inds]
            new_weights = old_weights_to_parents + old_weights_parents_to_grandparents
            new_weights = np.concatenate(new_weights.new_weights)
        elif reduction_method == 'resistance_distance':
            old_weights_to_parents = old_W_weight[sold_weights_to_parents_inds]
            old_weights_parents_to_grandparents = old_W_weights[old_weights_parents_to_grandparents_inds]
            new_weights = 1.0 / (1.0 / old_weights_to_parents + 1.0 / old_weights_parents_to_grandparents)
            new_weights = np.concatenate([new_weights, new_weights])
        else:
            raise ValueError('Unknown graph reduction method.')
        new_W = sparse.csc_matrix((new_weights, (i_inds, j_inds)), shape=(new_N, new_N))
        new_root = np.where(keep_inds == root)[0]
        parents = np.zeros(np.shape(keep_inds)[0], np.shape(keep_inds)[0])
        parents[:new_root - 1, new_root:] = new_non_root_parents
        depths = depths[keep_inds]
        depths = depths / 2.0
        Gtemp = graphs.Graph(new_W, coords=Gs[lev].coords[keep_inds], limits=G.limits, gtype='tree', root=new_root)
        if compute_full_eigen:
            Gs[lev + 1].compute_fourier_basis()
        Gs.append(Gtemp)
        old_W = new_W
        root = new_root
    return (Gs, subsampled_vertex_indices)