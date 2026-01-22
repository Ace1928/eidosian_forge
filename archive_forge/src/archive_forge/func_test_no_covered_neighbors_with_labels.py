import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
from networkx.algorithms.isomorphism.vf2pp import (
def test_no_covered_neighbors_with_labels(self):
    G1 = nx.DiGraph()
    G1.add_edges_from(self.G1_edges)
    G1.add_node(0)
    G2 = nx.relabel_nodes(G1, self.mapped)
    G1_degree = {n: (in_degree, out_degree) for (n, in_degree), (_, out_degree) in zip(G1.in_degree, G1.out_degree)}
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([self.mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    l1 = dict(G1.nodes(data='label', default=-1))
    l2 = dict(G2.nodes(data='label', default=-1))
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups({node: (in_degree, out_degree) for (node, in_degree), (_, out_degree) in zip(G2.in_degree(), G2.out_degree())}))
    m = {9: self.mapped[9], 1: self.mapped[1]}
    m_rev = {self.mapped[9]: 9, self.mapped[1]: 1}
    T1_out = {2, 4, 6}
    T1_in = {5, 7, 8}
    T1_tilde = {0, 3}
    T2_out = {'b', 'd', 'f'}
    T2_in = {'e', 'g', 'h'}
    T2_tilde = {'x', 'c'}
    sparams = _StateParameters(m, m_rev, T1_out, T1_in, T1_tilde, None, T2_out, T2_in, T2_tilde, None)
    u = 3
    candidates = _find_candidates_Di(u, gparams, sparams, G1_degree)
    assert candidates == {self.mapped[u]}
    u = 0
    candidates = _find_candidates_Di(u, gparams, sparams, G1_degree)
    assert candidates == {self.mapped[u]}
    G1.nodes[u]['label'] = 'blue'
    l1 = dict(G1.nodes(data='label', default=-1))
    l2 = dict(G2.nodes(data='label', default=-1))
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups({node: (in_degree, out_degree) for (node, in_degree), (_, out_degree) in zip(G2.in_degree(), G2.out_degree())}))
    candidates = _find_candidates_Di(u, gparams, sparams, G1_degree)
    assert candidates == set()
    m.pop(9)
    m_rev.pop(self.mapped[9])
    T1_out = {2, 4, 6}
    T1_in = {5}
    T1_tilde = {0, 3, 7, 8, 9}
    T2_out = {'b', 'd', 'f'}
    T2_in = {'e'}
    T2_tilde = {'x', 'c', 'g', 'h', 'i'}
    sparams = _StateParameters(m, m_rev, T1_out, T1_in, T1_tilde, None, T2_out, T2_in, T2_tilde, None)
    u = 7
    candidates = _find_candidates_Di(u, gparams, sparams, G1_degree)
    assert candidates == {self.mapped[u]}
    G1.nodes[8]['label'] = G1.nodes[7]['label']
    G2.nodes[self.mapped[8]]['label'] = G1.nodes[7]['label']
    l1 = dict(G1.nodes(data='label', default=-1))
    l2 = dict(G2.nodes(data='label', default=-1))
    gparams = _GraphParameters(G1, G2, l1, l2, nx.utils.groups(l1), nx.utils.groups(l2), nx.utils.groups({node: (in_degree, out_degree) for (node, in_degree), (_, out_degree) in zip(G2.in_degree(), G2.out_degree())}))
    candidates = _find_candidates_Di(u, gparams, sparams, G1_degree)
    assert candidates == {self.mapped[u], self.mapped[8]}