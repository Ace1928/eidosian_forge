from . import schema
from .jsonutil import get_column
from .search import Search
def rest_resource(self, name):
    resource_types = self._intf.inspect._resource_types(name)
    graph = nx.DiGraph()
    graph.add_node(name)
    graph.labels = {name: name}
    graph.weights = {name: 100.0}
    namespaces = set([exp.split(':')[0] for exp in resource_types])
    for ns in namespaces:
        graph.add_edge(name, ns)
        graph.weights[ns] = 70.0
        for exp in resource_types:
            if exp.startswith(ns):
                graph.add_edge(ns, exp)
                graph.weights[exp] = 40.0
    return graph