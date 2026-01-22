import pickle
from pyomo.common.dependencies import dill
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import _BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random
import pyomo.opt
def check_iteratively_adding_disjunctions_transform_model(self, transformation):
    model = ConcreteModel()
    model.x = Var(bounds=(-100, 100))
    model.disjunctionList = Disjunction(Any)
    model.obj = Objective(expr=model.x)
    for i in range(2):
        firstTermName = 'firstTerm[%s]' % i
        model.add_component(firstTermName, Disjunct())
        model.component(firstTermName).cons = Constraint(expr=model.x == 2 * i)
        secondTermName = 'secondTerm[%s]' % i
        model.add_component(secondTermName, Disjunct())
        model.component(secondTermName).cons = Constraint(expr=model.x >= i + 2)
        model.disjunctionList[i] = [model.component(firstTermName), model.component(secondTermName)]
        TransformationFactory('gdp.%s' % transformation).apply_to(model)
        if i == 0:
            self.check_first_iteration(model)
        if i == 1:
            self.check_second_iteration(model)