import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
def inactive_index_LP_c2_rule(model, i):
    if i == 1:
        return model.y >= -2
    else:
        return model.x <= 2