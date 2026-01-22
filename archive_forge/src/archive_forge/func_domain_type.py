from pyomo.common.modeling import NoArgumentGiven
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericValue, is_numeric_data, value
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.set_types import RealSet, IntegerSet
@domain_type.setter
def domain_type(self, domain_type):
    if domain_type not in IVariable._valid_domain_types:
        raise ValueError("Domain type '%s' is not valid. Must be one of: %s" % (self.domain_type, IVariable._valid_domain_types))
    self._domain_type = domain_type