import pytest
import networkx as nx
from networkx.algorithms.centrality.subgraph_alg import (
def test_subgraph_centrality_big_graph(self):
    g199 = nx.complete_graph(199)
    g200 = nx.complete_graph(200)
    comm199 = nx.subgraph_centrality(g199)
    comm199_exp = nx.subgraph_centrality_exp(g199)
    comm200 = nx.subgraph_centrality(g200)
    comm200_exp = nx.subgraph_centrality_exp(g200)