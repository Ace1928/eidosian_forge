import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
@nx._dispatch(edge_attrs={'attr': 'default', 'partition': 0}, preserve_edge_attrs='preserve_attrs')
def maximum_branching(G, attr='weight', default=1, preserve_attrs=False, partition=None):

    def edmonds_add_edge(G, edge_index, u, v, key, **d):
        """
        Adds an edge to `G` while also updating the edge index.

        This algorithm requires the use of an external dictionary to track
        the edge keys since it is possible that the source or destination
        node of an edge will be changed and the default key-handling
        capabilities of the MultiDiGraph class do not account for this.

        Parameters
        ----------
        G : MultiDiGraph
            The graph to insert an edge into.
        edge_index : dict
            A mapping from integers to the edges of the graph.
        u : node
            The source node of the new edge.
        v : node
            The destination node of the new edge.
        key : int
            The key to use from `edge_index`.
        d : keyword arguments, optional
            Other attributes to store on the new edge.
        """
        if key in edge_index:
            uu, vv, _ = edge_index[key]
            if u != uu or v != vv:
                raise Exception(f'Key {key!r} is already in use.')
        G.add_edge(u, v, key, **d)
        edge_index[key] = (u, v, G.succ[u][v][key])

    def edmonds_remove_node(G, edge_index, n):
        """
        Remove a node from the graph, updating the edge index to match.

        Parameters
        ----------
        G : MultiDiGraph
            The graph to remove an edge from.
        edge_index : dict
            A mapping from integers to the edges of the graph.
        n : node
            The node to remove from `G`.
        """
        keys = set()
        for keydict in G.pred[n].values():
            keys.update(keydict)
        for keydict in G.succ[n].values():
            keys.update(keydict)
        for key in keys:
            del edge_index[key]
        G.remove_node(n)
    candidate_attr = "edmonds' secret candidate attribute"
    new_node_base_name = 'edmonds new node base name '
    G_original = G
    G = nx.MultiDiGraph()
    G_edge_index = {}
    for key, (u, v, data) in enumerate(G_original.edges(data=True)):
        d = {attr: data.get(attr, default)}
        if data.get(partition) is not None:
            d[partition] = data.get(partition)
        if preserve_attrs:
            for d_k, d_v in data.items():
                if d_k != attr:
                    d[d_k] = d_v
        edmonds_add_edge(G, G_edge_index, u, v, key, **d)
    level = 0
    B = nx.MultiDiGraph()
    B_edge_index = {}
    graphs = []
    branchings = []
    selected_nodes = set()
    uf = nx.utils.UnionFind()
    circuits = []
    minedge_circuit = []

    def edmonds_find_desired_edge(v):
        """
        Find the edge directed towards v with maximal weight.

        If an edge partition exists in this graph, return the included
        edge if it exists and never return any excluded edge.

        Note: There can only be one included edge for each vertex otherwise
        the edge partition is empty.

        Parameters
        ----------
        v : node
            The node to search for the maximal weight incoming edge.
        """
        edge = None
        max_weight = -INF
        for u, _, key, data in G.in_edges(v, data=True, keys=True):
            if data.get(partition) == nx.EdgePartition.EXCLUDED:
                continue
            new_weight = data[attr]
            if data.get(partition) == nx.EdgePartition.INCLUDED:
                max_weight = new_weight
                edge = (u, v, key, new_weight, data)
                break
            if new_weight > max_weight:
                max_weight = new_weight
                edge = (u, v, key, new_weight, data)
        return (edge, max_weight)

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

    def is_root(G, u, edgekeys):
        """
        Returns True if `u` is a root node in G.

        Node `u` is a root node if its in-degree over the specified edges is zero.

        Parameters
        ----------
        G : Graph
            The current graph.
        u : node
            The node in `G` to check if it is a root.
        edgekeys : iterable of edges
            The edges for which to check if `u` is a root of.
        """
        if u not in G:
            raise Exception(f'{u!r} not in G')
        for v in G.pred[u]:
            for edgekey in G.pred[u][v]:
                if edgekey in edgekeys:
                    return (False, edgekey)
        else:
            return (True, None)
    nodes = iter(list(G.nodes))
    while True:
        try:
            v = next(nodes)
        except StopIteration:
            assert len(G) == len(B)
            if len(B):
                assert is_branching(B)
            graphs.append((G.copy(), G_edge_index.copy()))
            branchings.append((B.copy(), B_edge_index.copy()))
            circuits.append([])
            minedge_circuit.append(None)
            break
        else:
            if v in selected_nodes:
                continue
        selected_nodes.add(v)
        B.add_node(v)
        desired_edge, desired_edge_weight = edmonds_find_desired_edge(v)
        if desired_edge is not None and desired_edge_weight > 0:
            u = desired_edge[0]
            circuit = uf[u] == uf[v]
            dd = {attr: desired_edge_weight}
            if desired_edge[4].get(partition) is not None:
                dd[partition] = desired_edge[4].get(partition)
            edmonds_add_edge(B, B_edge_index, u, v, desired_edge[2], **dd)
            G[u][v][desired_edge[2]][candidate_attr] = True
            uf.union(u, v)
            if circuit:
                edmonds_step_I2(v, desired_edge, level)
                nodes = iter(list(G.nodes()))
                level += 1
    H = G_original.__class__()
    edges = set(branchings[level][1])
    while level > 0:
        level -= 1
        merged_node = new_node_base_name + str(level)
        circuit = circuits[level]
        isroot, edgekey = is_root(graphs[level + 1][0], merged_node, edges)
        edges.update(circuit)
        if isroot:
            minedge = minedge_circuit[level]
            if minedge is None:
                raise Exception
            edges.remove(minedge)
        else:
            G, G_edge_index = graphs[level]
            target = G_edge_index[edgekey][1]
            for edgekey in circuit:
                u, v, data = G_edge_index[edgekey]
                if v == target:
                    break
            else:
                raise Exception("Couldn't find edge incoming to merged node.")
            edges.remove(edgekey)
    H.add_nodes_from(G_original)
    for edgekey in edges:
        u, v, d = graphs[0][1][edgekey]
        dd = {attr: d[attr]}
        if preserve_attrs:
            for key, value in d.items():
                if key not in [attr, candidate_attr]:
                    dd[key] = value
        H.add_edge(u, v, **dd)
    return H