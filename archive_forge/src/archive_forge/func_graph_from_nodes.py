import unittest
from traits.observation._observer_graph import ObserverGraph
def graph_from_nodes(*nodes):
    nodes = nodes[::-1]
    graph = ObserverGraph(node=nodes[0])
    for node in nodes[1:]:
        graph = ObserverGraph(node=node, children=[graph])
    return graph