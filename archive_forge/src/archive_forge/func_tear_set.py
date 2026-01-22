from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def tear_set(self, G):
    key = 'tear_set'

    def fcn(G):
        tset = self.options[key]
        if tset is not None:
            arc_map = self.arc_to_edge(G)
            edge_map = self.edge_to_idx(G)
            res = []
            for arc in tset:
                res.append(edge_map[arc_map[arc]])
            if not self.check_tear_set(G, res):
                raise ValueError('Tear set found in options is insufficient to solve network')
            self.cache[key] = res
            return res
        method = self.options['select_tear_method']
        if method == 'mip':
            return self.select_tear_mip(G, self.options['tear_solver'], self.options['tear_solver_io'], self.options['tear_solver_options'])
        elif method == 'heuristic':
            return self.select_tear_heuristic(G)[0][0]
        else:
            raise ValueError("Invalid select_tear_method '%s'" % (method,))
    return self.cacher(key, fcn, G)