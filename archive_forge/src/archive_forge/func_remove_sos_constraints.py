import abc
from typing import List
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.contrib.solver.util import collect_vars_and_named_exprs, get_objective
def remove_sos_constraints(self, cons: List[_SOSConstraintData]):
    self._remove_sos_constraints(cons)
    for con in cons:
        if con not in self._vars_referenced_by_con:
            raise ValueError('cannot remove constraint {name} - it was not added'.format(name=con.name))
        for v in self._vars_referenced_by_con[con]:
            self._referenced_variables[id(v)][1].pop(con)
        self._check_to_remove_vars(self._vars_referenced_by_con[con])
        del self._active_constraints[con]
        del self._named_expressions[con]
        del self._vars_referenced_by_con[con]