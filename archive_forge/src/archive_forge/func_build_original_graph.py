import pytest
import networkx as nx
def build_original_graph(self):
    nodes = {'A': {'color': 'Red'}, 'B': {'color': 'Red'}, 'C': {'color': 'Green'}, 'D': {'color': 'Green'}, 'E': {'color': 'Blue'}, 'F': {'color': 'Blue'}, 'G': {'color': 'Yellow'}, 'H': {'color': 'Yellow'}}
    edges = [('A', 'C', ['Weak', 'Strong']), ('A', 'E', ['Strong']), ('A', 'F', ['Weak']), ('B', 'D', ['Weak', 'Strong']), ('B', 'E', ['Weak']), ('B', 'F', ['Strong']), ('C', 'G', ['Weak', 'Strong']), ('C', 'F', ['Strong']), ('D', 'E', ['Strong']), ('D', 'H', ['Weak', 'Strong']), ('G', 'E', ['Strong']), ('H', 'F', ['Strong'])]
    G = nx.MultiDiGraph()
    for node in nodes:
        attributes = nodes[node]
        G.add_node(node, **attributes)
    for source, target, types in edges:
        for type in types:
            G.add_edge(source, target, type=type)
    return G