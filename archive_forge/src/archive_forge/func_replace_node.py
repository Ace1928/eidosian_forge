import networkx as nx
from networkx.utils import not_implemented_for
def replace_node(self):
    adj_view = self.g[self.original]
    for outer, inner, (neighbor, edge_attrs) in zip(self.outer_vertices, self.inner_vertices, list(adj_view.items())):
        self.g.add_edge(outer, inner)
        self.g.add_edge(outer, neighbor, **edge_attrs)
    for core in self.core_vertices:
        for inner in self.inner_vertices:
            self.g.add_edge(core, inner)
    self.g.remove_node(self.original)