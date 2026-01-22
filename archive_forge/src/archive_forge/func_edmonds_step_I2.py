import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def edmonds_step_I2(v, desired_edge, level):
    """
        Perform step I2 from Edmonds' paper

        First, check if the last step I1 created a cycle. If it did not, do nothing.
        If it did, store the cycle for later reference and contract it.

        Parameters
        ----------
        v : node
            The current node to consider
        desired_edge : edge
            The minimum desired edge to remove from the cycle.
        level : int
            The current level, i.e. the number of cycles that have already been removed.
        """
    u = desired_edge[0]
    Q_nodes = nx.shortest_path(B, v, u)
    Q_edges = [list(B[Q_nodes[i]][vv].keys())[0] for i, vv in enumerate(Q_nodes[1:])]
    Q_edges.append(desired_edge[2])
    minweight = INF
    minedge = None
    Q_incoming_weight = {}
    for edge_key in Q_edges:
        u, v, data = B_edge_index[edge_key]
        w = data[attr]
        Q_incoming_weight[v] = w
        if data.get(partition) == nx.EdgePartition.INCLUDED:
            continue
        if w < minweight:
            minweight = w
            minedge = edge_key
    circuits.append(Q_edges)
    minedge_circuit.append(minedge)
    graphs.append((G.copy(), G_edge_index.copy()))
    branchings.append((B.copy(), B_edge_index.copy()))
    new_node = new_node_base_name + str(level)
    G.add_node(new_node)
    new_edges = []
    for u, v, key, data in G.edges(data=True, keys=True):
        if u in Q_incoming_weight:
            if v in Q_incoming_weight:
                continue
            else:
                dd = data.copy()
                new_edges.append((new_node, v, key, dd))
        elif v in Q_incoming_weight:
            w = data[attr]
            w += minweight - Q_incoming_weight[v]
            dd = data.copy()
            dd[attr] = w
            new_edges.append((u, new_node, key, dd))
        else:
            continue
    for node in Q_nodes:
        edmonds_remove_node(G, G_edge_index, node)
        edmonds_remove_node(B, B_edge_index, node)
    selected_nodes.difference_update(set(Q_nodes))
    for u, v, key, data in new_edges:
        edmonds_add_edge(G, G_edge_index, u, v, key, **data)
        if candidate_attr in data:
            del data[candidate_attr]
            edmonds_add_edge(B, B_edge_index, u, v, key, **data)
            uf.union(u, v)