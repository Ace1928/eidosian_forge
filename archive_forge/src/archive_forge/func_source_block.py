import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def source_block(b):
    b.p_out = Var(b.model().time)
    b.outlet = Port(initialize={'p': (b.p_out, Port.Extensive, {'include_splitfrac': False})})