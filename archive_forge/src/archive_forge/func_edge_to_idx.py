from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def edge_to_idx(self, G):
    """Returns a mapping from edges to indexes for a graph"""

    def fcn(G):
        res = dict()
        i = -1
        for edge in G.edges:
            i += 1
            res[edge] = i
        return res
    return self.cacher('edge_to_idx', fcn, G)