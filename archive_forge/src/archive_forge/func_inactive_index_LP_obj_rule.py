import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
def inactive_index_LP_obj_rule(model, i):
    if i == 1:
        return model.x - model.y
    else:
        return -model.x + model.y + model.z