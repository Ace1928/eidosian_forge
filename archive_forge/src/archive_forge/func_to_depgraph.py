from itertools import chain
from nltk.internals import Counter
def to_depgraph(self, rel=None):
    from nltk.parse.dependencygraph import DependencyGraph
    depgraph = DependencyGraph()
    nodes = depgraph.nodes
    self._to_depgraph(nodes, 0, 'ROOT')
    for address, node in nodes.items():
        for n2 in (n for n in nodes.values() if n['rel'] != 'TOP'):
            if n2['head'] == address:
                relation = n2['rel']
                node['deps'].setdefault(relation, [])
                node['deps'][relation].append(n2['address'])
    depgraph.root = nodes[1]
    return depgraph