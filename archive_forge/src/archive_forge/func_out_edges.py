from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.digraph import DiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.classes.reportviews import (
from networkx.exception import NetworkXError
@cached_property
def out_edges(self):
    return OutMultiEdgeView(self)