from a triple store that implements the Redland storage interface; similarly,
import os
from io import StringIO
from Bio import MissingPythonDependencyError
from Bio.Phylo import CDAO
from ._cdao_owl import cdao_namespaces, resolve_uri
def parse_handle_to_graph(self, rooted=False, parse_format='turtle', context=None, **kwargs):
    """Parse self.handle into RDF model self.model."""
    if self.graph is None:
        self.graph = rdflib.Graph()
    graph = self.graph
    for k, v in RDF_NAMESPACES.items():
        graph.bind(k, v)
    self.rooted = rooted
    if 'base_uri' in kwargs:
        base_uri = kwargs['base_uri']
    else:
        base_uri = 'file://' + os.path.abspath(self.handle.name).replace('\\', '/')
    graph.parse(file=self.handle, publicID=base_uri, format=parse_format)
    return self.parse_graph(graph, context=context)