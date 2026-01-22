from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def indexes_to_arcs(self, G, lst):
    """
        Converts a list of edge indexes to the corresponding Arcs

        Arguments
        ---------
            G
                A networkx graph corresponding to lst
            lst
                A list of edge indexes to convert to tuples

        Returns:
            A list of arcs
        """
    edge_list = self.idx_to_edge(G)
    res = []
    for ei in lst:
        edge = edge_list[ei]
        res.append(G.edges[edge]['arc'])
    return res