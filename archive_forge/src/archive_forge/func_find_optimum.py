import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def find_optimum(self, attr='weight', default=1, kind='max', style='branching', preserve_attrs=False, partition=None, seed=None):
    """
        Returns a branching from G.

        Parameters
        ----------
        attr : str
            The edge attribute used to in determining optimality.
        default : float
            The value of the edge attribute used if an edge does not have
            the attribute `attr`.
        kind : {'min', 'max'}
            The type of optimum to search for, either 'min' or 'max'.
        style : {'branching', 'arborescence'}
            If 'branching', then an optimal branching is found. If `style` is
            'arborescence', then a branching is found, such that if the
            branching is also an arborescence, then the branching is an
            optimal spanning arborescences. A given graph G need not have
            an optimal spanning arborescence.
        preserve_attrs : bool
            If True, preserve the other edge attributes of the original
            graph (that are not the one passed to `attr`)
        partition : str
            The edge attribute holding edge partition data. Used in the
            spanning arborescence iterator.
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        H : (multi)digraph
            The branching.

        """
    self._init(attr, default, kind, style, preserve_attrs, seed, partition)
    uf = self.uf
    G, B = (self.G, self.B)
    D = set()
    nodes = iter(list(G.nodes()))
    attr = self._attr
    G_pred = G.pred

    def desired_edge(v):
        """
            Find the edge directed toward v with maximal weight.

            If an edge partition exists in this graph, return the included edge
            if it exists and no not return any excluded edges. There can only
            be one included edge for each vertex otherwise the edge partition is
            empty.
            """
        edge = None
        weight = -INF
        for u, _, key, data in G.in_edges(v, data=True, keys=True):
            if data.get(partition) == nx.EdgePartition.EXCLUDED:
                continue
            new_weight = data[attr]
            if data.get(partition) == nx.EdgePartition.INCLUDED:
                weight = new_weight
                edge = (u, v, key, new_weight, data)
                return (edge, weight)
            if new_weight > weight:
                weight = new_weight
                edge = (u, v, key, new_weight, data)
        return (edge, weight)
    while True:
        try:
            v = next(nodes)
        except StopIteration:
            assert len(G) == len(B)
            if len(B):
                assert is_branching(B)
            if self.store:
                self.graphs.append(G.copy())
                self.branchings.append(B.copy())
                self.circuits.append([])
                self.minedge_circuit.append(None)
            break
        else:
            if v in D:
                continue
        D.add(v)
        B.add_node(v)
        edge, weight = desired_edge(v)
        if edge is None:
            continue
        else:
            u = edge[0]
            if uf[u] == uf[v]:
                Q_nodes, Q_edges = get_path(B, v, u)
                Q_edges.append(edge[2])
            else:
                Q_nodes, Q_edges = (None, None)
            if self.style == 'branching' and weight <= 0:
                acceptable = False
            else:
                acceptable = True
            if acceptable:
                dd = {attr: weight}
                if edge[4].get(partition) is not None:
                    dd[partition] = edge[4].get(partition)
                B.add_edge(u, v, edge[2], **dd)
                G[u][v][edge[2]][self.candidate_attr] = True
                uf.union(u, v)
                if Q_edges is not None:
                    minweight = INF
                    minedge = None
                    Q_incoming_weight = {}
                    for edge_key in Q_edges:
                        u, v, data = B.edge_index[edge_key]
                        w = data[attr]
                        Q_incoming_weight[v] = w
                        if data.get(partition) == nx.EdgePartition.INCLUDED:
                            continue
                        if w < minweight:
                            minweight = w
                            minedge = edge_key
                    self.circuits.append(Q_edges)
                    self.minedge_circuit.append(minedge)
                    if self.store:
                        self.graphs.append(G.copy())
                    self.branchings.append(B.copy())
                    new_node = self.template.format(self.level)
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
                    G.remove_nodes_from(Q_nodes)
                    B.remove_nodes_from(Q_nodes)
                    D.difference_update(set(Q_nodes))
                    for u, v, key, data in new_edges:
                        G.add_edge(u, v, key, **data)
                        if self.candidate_attr in data:
                            del data[self.candidate_attr]
                            B.add_edge(u, v, key, **data)
                            uf.union(u, v)
                    nodes = iter(list(G.nodes()))
                    self.level += 1
    H = self.G_original.__class__()

    def is_root(G, u, edgekeys):
        """
            Returns True if `u` is a root node in G.

            Node `u` will be a root node if its in-degree, restricted to the
            specified edges, is equal to 0.

            """
        if u not in G:
            raise Exception(f'{u!r} not in G')
        for v in G.pred[u]:
            for edgekey in G.pred[u][v]:
                if edgekey in edgekeys:
                    return (False, edgekey)
        else:
            return (True, None)
    edges = set(self.branchings[self.level].edge_index)
    while self.level > 0:
        self.level -= 1
        merged_node = self.template.format(self.level)
        circuit = self.circuits[self.level]
        isroot, edgekey = is_root(self.graphs[self.level + 1], merged_node, edges)
        edges.update(circuit)
        if isroot:
            minedge = self.minedge_circuit[self.level]
            if minedge is None:
                raise Exception
            edges.remove(minedge)
        else:
            G = self.graphs[self.level]
            target = G.edge_index[edgekey][1]
            for edgekey in circuit:
                u, v, data = G.edge_index[edgekey]
                if v == target:
                    break
            else:
                raise Exception("Couldn't find edge incoming to merged node.")
            edges.remove(edgekey)
    self.edges = edges
    H.add_nodes_from(self.G_original)
    for edgekey in edges:
        u, v, d = self.graphs[0].edge_index[edgekey]
        dd = {self.attr: self.trans(d[self.attr])}
        if preserve_attrs:
            for key, value in d.items():
                if key not in [self.attr, self.candidate_attr]:
                    dd[key] = value
        H.add_edge(u, v, **dd)
    return H