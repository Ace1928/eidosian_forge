from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state
def prepare_graph():
    """
        For the graph `G`, remove all edges not in the set `V` and then
        contract all edges in the set `U`.

        Returns
        -------
        A copy of `G` which has had all edges not in `V` removed and all edges
        in `U` contracted.
        """
    result = nx.MultiGraph(incoming_graph_data=G)
    edges_to_remove = set(result.edges()).difference(V)
    result.remove_edges_from(edges_to_remove)
    merged_nodes = {}
    for u, v in U:
        u_rep = find_node(merged_nodes, u)
        v_rep = find_node(merged_nodes, v)
        if u_rep == v_rep:
            continue
        nx.contracted_nodes(result, u_rep, v_rep, self_loops=False, copy=False)
        merged_nodes[v_rep] = u_rep
    return (merged_nodes, result)