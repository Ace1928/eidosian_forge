from pyomo.environ import (
def constr_rule(model):
    return (model.a, model.y * model.y, None)