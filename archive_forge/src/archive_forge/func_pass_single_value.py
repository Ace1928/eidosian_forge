from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def pass_single_value(self, port, name, member, val, fixed):
    """
        Fix the value of the port member and add it to the fixed set.
        If the member is an expression, appropriately fix the value of
        its free variable. Error if the member is already fixed but
        different from val, or if the member has more than one free
        variable."
        """
    eq_tol = self.options['almost_equal_tol']
    if member.is_fixed():
        if abs(value(member) - val) > eq_tol:
            raise RuntimeError("Member '%s' of port '%s' is already fixed but has a different value (by > %s) than what is being passed to it" % (name, port.name, eq_tol))
    elif member.is_expression_type():
        repn = generate_standard_repn(member - val)
        if repn.is_linear() and len(repn.linear_vars) == 1:
            fval = (0 - repn.constant) / repn.linear_coefs[0]
            var = repn.linear_vars[0]
            fixed.add(var)
            var.fix(float(fval))
        else:
            raise RuntimeError("Member '%s' of port '%s' had more than one free variable when trying to pass a value to it. Please fix more variables before passing to this port." % (name, port.name))
    else:
        fixed.add(member)
        member.fix(float(val))