import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
@unittest.skipIf(not ipopt_available, 'ipopt solver not available')
def test_fixed_disjuncts(self):
    self._test_disjuncts(True)
    self._test_disjuncts(False)