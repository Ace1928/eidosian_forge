import os.path as op
import numpy as np
import networkx as nx
import pickle
from ... import logging
from ..base import (
from .base import have_cv
def ntwks_to_matrices(in_files, edge_key):
    first = _read_pickle(in_files[0])
    files = len(in_files)
    nodes = len(first.nodes())
    matrix = np.zeros((nodes, nodes, files))
    for idx, name in enumerate(in_files):
        graph = _read_pickle(name)
        for u, v, d in graph.edges(data=True):
            try:
                graph[u][v]['weight'] = d[edge_key]
            except:
                raise KeyError('the graph edges do not have {} attribute'.format(edge_key))
        matrix[:, :, idx] = nx.to_numpy_array(graph)
    return matrix