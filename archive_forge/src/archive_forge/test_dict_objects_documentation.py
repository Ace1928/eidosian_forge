import pyomo.common.unittest as unittest
from pyomo.core.base import ConcreteModel, Var, Reals
from pyomo.core.beta.dict_objects import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData

    # make sure an existing Data object is NOT replaced
    # by a call to setitem but simply updated.
    def test_setitem_exists(self):
        model = self.model
        index = ['a', 1, None, (1,), (1,2)]
        model.c = self._ctype((i, self._arg()) for i in index)
        self.assertEqual(len(model.c), len(index))
        for i in index:
            self.assertTrue(i in model.c)
            cdata = model.c[i]
            model.c[i] = self._arg()
            self.assertEqual(len(model.c), len(index))
            self.assertTrue(i in model.c)
            self.assertEqual(id(cdata), id(model.c[i]))
    