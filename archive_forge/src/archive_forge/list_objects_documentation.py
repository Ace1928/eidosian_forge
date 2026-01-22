import logging
from weakref import ref as weakref_ref
from pyomo.common.log import is_debug_set
from pyomo.core.base.set_types import Any
from pyomo.core.base.var import IndexedVar, _VarData
from pyomo.core.base.constraint import IndexedConstraint, _ConstraintData
from pyomo.core.base.objective import IndexedObjective, _ObjectiveData
from pyomo.core.base.expression import IndexedExpression, _ExpressionData
from collections.abc import MutableSequence
S.reverse() -- reverse *IN PLACE*