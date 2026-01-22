from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def select_tear_mip_model(self, G):
    """
        Generate a model for selecting tears from the given graph

        Returns
        -------
            model
            bin_list
                A list of the binary variables representing each edge,
                indexed by the edge index of the graph
        """
    model = ConcreteModel()
    bin_list = []
    for i in range(G.number_of_edges()):
        vname = 'edge%s' % i
        var = Var(domain=Binary)
        bin_list.append(var)
        model.add_component(vname, var)
    mct = model.max_cycle_tears = Var()
    _, cycleEdges = self.all_cycles(G)
    for i in range(len(cycleEdges)):
        ecyc = cycleEdges[i]
        ename = 'cycle_sum%s' % i
        expr = Expression(expr=sum((bin_list[i] for i in ecyc)))
        model.add_component(ename, expr)
        cname_min = 'cycle_min%s' % i
        con_min = Constraint(expr=expr >= 1)
        model.add_component(cname_min, con_min)
        cname_mct = mct.name + '_geq%s' % i
        con_mct = Constraint(expr=mct >= expr)
        model.add_component(cname_mct, con_mct)
    obj_expr = 1000 * mct + sum((var for var in bin_list))
    model.obj = Objective(expr=obj_expr, sense=minimize)
    return (model, bin_list)