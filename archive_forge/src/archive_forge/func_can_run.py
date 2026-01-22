import networkx as nx
from networkx import DiGraph, Graph, MultiDiGraph, MultiGraph, PlanarEmbedding
from networkx.classes.reportviews import NodeView
def can_run(self, name, args, kwargs):
    return hasattr(self, name)