from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def check_value_fix(self, port, var, default, fixed, use_guesses, extensive=False):
    """
        Try to fix the var at its current value or the default, else error
        """
    val = None
    if var.value is not None:
        val = var.value
    elif default is not None:
        val = default
    if val is None:
        raise RuntimeError("Encountered a free inlet %svariable '%s' %s port '%s' with no %scurrent value, or default_guess option, while attempting to compute the unit." % ('extensive ' if extensive else '', var.name, ('on', 'to')[int(extensive)], port.name, 'guess, ' if use_guesses else ''))
    fixed.add(var)
    var.fix(float(val))