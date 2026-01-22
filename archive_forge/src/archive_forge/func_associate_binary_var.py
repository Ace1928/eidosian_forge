import logging
from weakref import ref as weakref_ref, ReferenceType
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set, BooleanSet, Binary
from pyomo.core.base.util import is_functor
from pyomo.core.base.var import Var
def associate_binary_var(self, binary_var):
    """Associate a binary _VarData to this _GeneralBooleanVarData"""
    if self._associated_binary is not None and type(self._associated_binary) is not _DeprecatedImplicitAssociatedBinaryVariable:
        raise RuntimeError("Reassociating BooleanVar '%s' (currently associated with '%s') with '%s' is not allowed" % (self.name, self._associated_binary().name if self._associated_binary is not None else None, binary_var.name if binary_var is not None else None))
    if binary_var is not None:
        self._associated_binary = weakref_ref(binary_var)