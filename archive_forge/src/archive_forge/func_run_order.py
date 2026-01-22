from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def run_order(self, G, order, function, ignore=None, use_guesses=False):
    """
        Run computations in the order provided by calling the function

        Arguments
        ---------
            G
                A networkx graph corresponding to order
            order
                The order in which to run each node in the graph
            function
                The function to be called on each block/node
            ignore
                Edge indexes to ignore when passing values
            use_guesses
                If True, will check the guesses dict when fixing
                free variables before calling function
        """
    fixed_inputs = self.fixed_inputs()
    fixed_outputs = ComponentSet()
    edge_map = self.edge_to_idx(G)
    arc_map = self.arc_to_edge(G)
    guesses = self.options['guesses']
    default = self.options['default_guess']
    for lev in order:
        for unit in lev:
            if unit not in fixed_inputs:
                fixed_inputs[unit] = ComponentSet()
            fixed_ins = fixed_inputs[unit]
            for port in unit.component_data_objects(Port):
                if not len(port.sources()):
                    continue
                if use_guesses and port in guesses:
                    self.load_guesses(guesses, port, fixed_ins)
                self.load_values(port, default, fixed_ins, use_guesses)
            function(unit)
            for var in fixed_ins:
                var.free()
            fixed_ins.clear()
            for port in unit.component_data_objects(Port):
                dests = port.dests()
                if not len(dests):
                    continue
                for var in port.iter_vars(expr_vars=True, fixed=False):
                    fixed_outputs.add(var)
                    var.fix()
                for arc in dests:
                    if arc in arc_map and edge_map[arc_map[arc]] not in ignore:
                        self.pass_values(arc, fixed_inputs)
                for var in fixed_outputs:
                    var.free()
                fixed_outputs.clear()