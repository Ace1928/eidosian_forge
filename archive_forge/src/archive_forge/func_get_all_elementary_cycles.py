from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
def get_all_elementary_cycles(graph):
    """

    Args:
        graph:
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError('graph should be a networkx Graph object.')
    cycle_basis = nx.cycle_basis(graph)
    if len(cycle_basis) < 2:
        return {SimpleGraphCycle(c) for c in cycle_basis}
    all_edges_dict = {}
    index2edge = []
    edge_idx = 0
    for n1, n2 in graph.edges:
        all_edges_dict[n1, n2] = edge_idx
        all_edges_dict[n2, n1] = edge_idx
        index2edge.append((n1, n2))
        edge_idx += 1
    cycles_matrix = np.zeros(shape=(len(cycle_basis), edge_idx), dtype=bool)
    for icycle, cycle in enumerate(cycle_basis):
        for in1, n1 in enumerate(cycle, start=1):
            n2 = cycle[in1 % len(cycle)]
            iedge = all_edges_dict[n1, n2]
            cycles_matrix[icycle, iedge] = True
    elementary_cycles_list = []
    for cycle_idx in range(1, len(cycle_basis) + 1):
        for cycles_combination in itertools.combinations(cycles_matrix, cycle_idx):
            edges_counts = np.array(np.mod(np.sum(cycles_combination, axis=0), 2), dtype=bool)
            edges = [edge for iedge, edge in enumerate(index2edge) if edges_counts[iedge]]
            try:
                sgc = SimpleGraphCycle.from_edges(edges, edges_are_ordered=False)
            except ValueError as ve:
                msg = ve.args[0]
                if msg == 'SimpleGraphCycle is not valid : Duplicate nodes.':
                    continue
                if msg == 'Could not construct a cycle from edges.':
                    continue
                raise
            elementary_cycles_list.append(sgc)
    return elementary_cycles_list