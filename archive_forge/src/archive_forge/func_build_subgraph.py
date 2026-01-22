import math
import sys
from Bio import MissingPythonDependencyError
def build_subgraph(graph, top):
    """Walk down the Tree, building graphs, edges and nodes."""
    for clade in top:
        graph.add_node(clade.root)
        add_edge(graph, top.root, clade.root)
        build_subgraph(graph, clade)