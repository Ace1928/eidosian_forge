import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr.numvalue import (
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.variable import (
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import RealSet, IntegerSet, BooleanSet
from pyomo.core.base.set import (
def test_extract_domain_type_and_bounds(self):
    domain_type, lb, ub = _extract_domain_type_and_bounds(None, None, None, None)
    self.assertIs(domain_type, RealSet)
    self.assertIs(lb, None)
    self.assertIs(ub, None)