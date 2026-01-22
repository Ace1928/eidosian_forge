import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def generate_random_paths(G, sample_size, path_length=5, index_map=None, weight='weight'):
    """Randomly generate `sample_size` paths of length `path_length`.

    Parameters
    ----------
    G : NetworkX graph
        A NetworkX graph
    sample_size : integer
        The number of paths to generate. This is ``R`` in [1]_.
    path_length : integer (default = 5)
        The maximum size of the path to randomly generate.
        This is ``T`` in [1]_. According to the paper, ``T >= 5`` is
        recommended.
    index_map : dictionary, optional
        If provided, this will be populated with the inverted
        index of nodes mapped to the set of generated random path
        indices within ``paths``.
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.

    Returns
    -------
    paths : generator of lists
        Generator of `sample_size` paths each with length `path_length`.

    Examples
    --------
    Note that the return value is the list of paths:

    >>> G = nx.star_graph(3)
    >>> random_path = nx.generate_random_paths(G, 2)

    By passing a dictionary into `index_map`, it will build an
    inverted index mapping of nodes to the paths in which that node is present:

    >>> G = nx.star_graph(3)
    >>> index_map = {}
    >>> random_path = nx.generate_random_paths(G, 3, index_map=index_map)
    >>> paths_containing_node_0 = [random_path[path_idx] for path_idx in index_map.get(0, [])]

    References
    ----------
    .. [1] Zhang, J., Tang, J., Ma, C., Tong, H., Jing, Y., & Li, J.
           Panther: Fast top-k similarity search on large networks.
           In Proceedings of the ACM SIGKDD International Conference
           on Knowledge Discovery and Data Mining (Vol. 2015-August, pp. 1445–1454).
           Association for Computing Machinery. https://doi.org/10.1145/2783258.2783267.
    """
    import numpy as np
    adj_mat = nx.to_numpy_array(G, weight=weight)
    inv_row_sums = np.reciprocal(adj_mat.sum(axis=1)).reshape(-1, 1)
    transition_probabilities = adj_mat * inv_row_sums
    node_map = np.array(G)
    num_nodes = G.number_of_nodes()
    for path_index in range(sample_size):
        node_index = np.random.randint(0, high=num_nodes)
        node = node_map[node_index]
        path = [node]
        if index_map is not None:
            if node in index_map:
                index_map[node].add(path_index)
            else:
                index_map[node] = {path_index}
        starting_index = node_index
        for _ in range(path_length):
            neighbor_index = np.random.choice(num_nodes, p=transition_probabilities[starting_index])
            starting_index = neighbor_index
            neighbor_node = node_map[neighbor_index]
            path.append(neighbor_node)
            if index_map is not None:
                if neighbor_node in index_map:
                    index_map[neighbor_node].add(path_index)
                else:
                    index_map[neighbor_node] = {path_index}
        yield path