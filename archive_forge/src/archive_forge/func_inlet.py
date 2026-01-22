import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
@m.prod.Port()
def inlet(b):
    return dict(flow=(b.flow_in, Port.Extensive), mass=(b.mass_in, Port.Extensive), temperature=b.temperature_in, pressure=b.pressure_in, expr_idx=(b.actual_var_idx_in, Port.Extensive), expr=(b.actual_var_in, Port.Extensive))