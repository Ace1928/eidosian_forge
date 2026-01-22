import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def initialize_mixer(self):
    for i in self.flow_out:
        self.flow_out[i].value = value(self.flow_in_side_1[i] + self.flow_in_side_2[i])
    for i in self.expr_var_idx_out:
        self.expr_var_idx_out[i].value = value(self.expr_var_idx_in_side_1[i] + self.expr_var_idx_in_side_2[i])
    self.expr_var_out.value = value(self.expr_var_in_side_1 + self.expr_var_in_side_2)
    assert self.temperature_in_side_1.value == self.temperature_in_side_2.value
    self.temperature_out.value = value(self.temperature_in_side_1)
    assert self.pressure_in_side_1.value == self.pressure_in_side_2.value
    self.pressure_out.value = value(self.pressure_in_side_1)