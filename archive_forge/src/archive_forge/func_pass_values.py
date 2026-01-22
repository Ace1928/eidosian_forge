from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def pass_values(self, arc, fixed_inputs):
    """
        Pass the values from one unit to the next, recording only those that
        were not already fixed in the provided dict that maps blocks to sets.
        """
    eblock = arc.expanded_block
    src, dest = (arc.src, arc.dest)
    dest_unit = dest.parent_block()
    eq_tol = self.options['almost_equal_tol']
    if dest_unit not in fixed_inputs:
        fixed_inputs[dest_unit] = ComponentSet()
    sf = eblock.component('splitfrac')
    if sf is not None and (not sf.is_fixed()):
        if sf.value is not None:
            fixed_inputs[dest_unit].add(sf)
            sf.fix()
        else:
            raise RuntimeError("Found free splitfrac for arc '%s' with no current value. Please use the set_split_fraction method on its source port to set this value before expansion, or set its value manually if expansion has already occurred." % arc.name)
    elif sf is None:
        for name, mem in src.vars.items():
            if not src.is_extensive(name):
                continue
            evar = eblock.component(name)
            if evar is None:
                continue
            if len(src.dests()) > 1:
                raise Exception("This still needs to be figured out (arc '%s')" % arc.name)
            if mem.is_indexed():
                evars = [(evar[i], i) for i in evar]
            else:
                evars = [(evar, None)]
            for evar, idx in evars:
                fixed_inputs[dest_unit].add(evar)
                val = value(mem[idx] if mem.is_indexed() else mem)
                evar.fix(float(val))
    for con in eblock.component_data_objects(Constraint, active=True):
        if not con.equality:
            raise RuntimeError("Found inequality constraint '%s'. Please do not modify the expanded block." % con.name)
        repn = generate_standard_repn(con.body)
        if repn.is_fixed():
            if abs(value(con.lower) - repn.constant) > eq_tol:
                raise RuntimeError("Found connected ports '%s' and '%s' both with fixed but different values (by > %s) for constraint '%s'" % (src, dest, eq_tol, con.name))
            continue
        if not (repn.is_linear() and len(repn.linear_vars) == 1):
            raise RuntimeError("Constraint '%s' had more than one free variable when trying to pass a value to its destination. Please fix more variables before passing across this arc." % con.name)
        val = (value(con.lower) - repn.constant) / repn.linear_coefs[0]
        var = repn.linear_vars[0]
        fixed_inputs[dest_unit].add(var)
        var.fix(float(val))