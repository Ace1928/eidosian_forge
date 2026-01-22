from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def compute_err(self, svals, dvals, tol_type):
    """Compute the diff between svals and dvals for the given tol_type"""
    if tol_type not in ('abs', 'rel'):
        raise ValueError("Invalid tol_type '%s'" % (tol_type,))
    diff = svals - dvals
    if tol_type == 'abs':
        err = diff
    else:
        old_settings = numpy.seterr(divide='ignore', invalid='ignore')
        err = diff / svals
        numpy.seterr(**old_settings)
        err[numpy.isnan(err)] = 0
        if any(numpy.isinf(err)):
            for i in range(len(err)):
                if numpy.isinf(err[i]):
                    err[i] = diff[i]
    return err