import pyomo.common.unittest as unittest
from pyomo.core.base import ConcreteModel, Var, Reals
from pyomo.core.beta.list_objects import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData
def test_model_clone(self):
    model = self.model
    index = range(5)
    model.c = self._ctype((self._cdatatype(self._arg()) for i in index))
    inst = model.clone()
    self.assertNotEqual(id(inst.c), id(model.c))
    for i in index:
        self.assertNotEqual(id(inst.c[i]), id(model.c[i]))