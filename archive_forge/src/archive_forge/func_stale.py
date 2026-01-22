from pyomo.common.modeling import NoArgumentGiven
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericValue, is_numeric_data, value
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.set_types import RealSet, IntegerSet
@stale.setter
def stale(self, stale):
    if stale:
        self._stale = 0
    else:
        self._stale = StaleFlagManager.get_flag(0)