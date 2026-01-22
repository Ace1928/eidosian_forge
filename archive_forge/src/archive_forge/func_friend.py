import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
@m.Arc(m.s)
def friend(m, i):
    return dict(source=m.prt1[i], destination=m.prt2[i])