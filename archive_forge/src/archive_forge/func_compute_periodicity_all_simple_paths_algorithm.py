from __future__ import annotations
import itertools
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from monty.json import MSONable, jsanitize
from networkx.algorithms.components import is_connected
from networkx.algorithms.traversal import bfs_tree
from pymatgen.analysis.chemenv.connectivity.environment_nodes import EnvironmentNode
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.graph_utils import get_delta
from pymatgen.analysis.chemenv.utils.math_utils import get_linearly_independent_vectors
def compute_periodicity_all_simple_paths_algorithm(self):
    """Get the periodicity vectors of the connected component."""
    self_loop_nodes = list(nx.nodes_with_selfloops(self._connected_subgraph))
    all_nodes_independent_cell_image_vectors = []
    simple_graph = nx.Graph(self._connected_subgraph)
    for test_node in self._connected_subgraph.nodes():
        this_node_cell_img_vectors = []
        if test_node in self_loop_nodes:
            for edge_data in self._connected_subgraph[test_node][test_node].values():
                if edge_data['delta'] == (0, 0, 0):
                    raise ValueError('There should not be self loops with delta image = (0, 0, 0).')
                this_node_cell_img_vectors.append(edge_data['delta'])
        for d1, d2 in itertools.combinations(this_node_cell_img_vectors, 2):
            if d1 == d2 or d1 == tuple((-ii for ii in d2)):
                raise ValueError('There should not be self loops with the same (or opposite) delta image.')
        this_node_cell_img_vectors = get_linearly_independent_vectors(this_node_cell_img_vectors)
        paths = []
        test_node_neighbors = simple_graph.neighbors(test_node)
        break_node_loop = False
        for test_node_neighbor in test_node_neighbors:
            if len(self._connected_subgraph[test_node][test_node_neighbor]) > 1:
                this_path_deltas = []
                node_node_neighbor_edges_data = list(self._connected_subgraph[test_node][test_node_neighbor].values())
                for edge1_data, edge2_data in itertools.combinations(node_node_neighbor_edges_data, 2):
                    delta1 = get_delta(test_node, test_node_neighbor, edge1_data)
                    delta2 = get_delta(test_node_neighbor, test_node, edge2_data)
                    this_path_deltas.append(delta1 + delta2)
                this_node_cell_img_vectors.extend(this_path_deltas)
                this_node_cell_img_vectors = get_linearly_independent_vectors(this_node_cell_img_vectors)
                if len(this_node_cell_img_vectors) == 3:
                    break
            for path in nx.all_simple_paths(simple_graph, test_node, test_node_neighbor, cutoff=len(self._connected_subgraph)):
                path_indices = [node_path.isite for node_path in path]
                if path_indices == [test_node.isite, test_node_neighbor.isite]:
                    continue
                path_indices.append(test_node.isite)
                path_indices = tuple(path_indices)
                if path_indices not in paths:
                    paths.append(path_indices)
                else:
                    continue
                path.append(test_node)
                this_path_deltas = [np.zeros(3, int)]
                for node1, node2 in [(node1, path[inode1 + 1]) for inode1, node1 in enumerate(path[:-1])]:
                    this_path_deltas_new = []
                    for edge_data in self._connected_subgraph[node1][node2].values():
                        delta = get_delta(node1, node2, edge_data)
                        for current_delta in this_path_deltas:
                            this_path_deltas_new.append(current_delta + delta)
                    this_path_deltas = this_path_deltas_new
                this_node_cell_img_vectors.extend(this_path_deltas)
                this_node_cell_img_vectors = get_linearly_independent_vectors(this_node_cell_img_vectors)
                if len(this_node_cell_img_vectors) == 3:
                    break_node_loop = True
                    break
            if break_node_loop:
                break
        this_node_cell_img_vectors = get_linearly_independent_vectors(this_node_cell_img_vectors)
        independent_cell_img_vectors = this_node_cell_img_vectors
        all_nodes_independent_cell_image_vectors.append(independent_cell_img_vectors)
        if len(independent_cell_img_vectors) == 3:
            break
    self._periodicity_vectors = []
    if len(all_nodes_independent_cell_image_vectors) != 0:
        for independent_cell_img_vectors in all_nodes_independent_cell_image_vectors:
            if len(independent_cell_img_vectors) > len(self._periodicity_vectors):
                self._periodicity_vectors = independent_cell_img_vectors
            if len(self._periodicity_vectors) == 3:
                break