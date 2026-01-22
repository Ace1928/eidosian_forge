from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def generate_first_x(self, G, tears):
    edge_list = self.idx_to_edge(G)
    x = []
    for tear in tears:
        arc = G.edges[edge_list[tear]]['arc']
        for name, index, mem in arc.src.iter_vars(names=True):
            peer = self.source_dest_peer(arc, name, index)
            x.append(value(peer))
    x = numpy.array(x)
    return x