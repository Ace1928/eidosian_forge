import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def test_graph_from_pr_2053():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('A', 'D'), ('A', 'F'), ('A', 'G'), ('B', 'C'), ('B', 'D'), ('B', 'G'), ('C', 'D'), ('C', 'E'), ('C', 'Z'), ('D', 'E'), ('D', 'F'), ('E', 'F'), ('E', 'Z'), ('F', 'Z'), ('G', 'Z')])
    for flow_func in flow_funcs:
        kwargs = {'flow_func': flow_func}
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        edge_paths = list(nx.edge_disjoint_paths(G, 'A', 'Z', **kwargs))
        assert are_edge_disjoint_paths(G, edge_paths), errmsg
        assert nx.edge_connectivity(G, 'A', 'Z') == len(edge_paths), errmsg
        node_paths = list(nx.node_disjoint_paths(G, 'A', 'Z', **kwargs))
        assert are_node_disjoint_paths(G, node_paths), errmsg
        assert nx.node_connectivity(G, 'A', 'Z') == len(node_paths), errmsg