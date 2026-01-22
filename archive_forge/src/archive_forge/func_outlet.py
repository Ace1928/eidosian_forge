import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def outlet(b):
    return dict(flow=(b.flow_out, Port.Extensive), mass=(b.mass_out, Port.Extensive), temperature=b.temperature_out, pressure=b.pressure_out, expr_idx=(b.expr_idx_out, Port.Extensive), expr=(b.expr_out, Port.Extensive))