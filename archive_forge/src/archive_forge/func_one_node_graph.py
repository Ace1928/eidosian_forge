import itertools
import pytest
import networkx as nx
def one_node_graph():
    graph = nx.Graph()
    graph.add_nodes_from([1])
    return graph