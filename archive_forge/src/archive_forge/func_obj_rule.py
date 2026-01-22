from pyomo.environ import AbstractModel, RangeSet, Var, Objective, sum_product
def obj_rule(model):
    return sum_product(model.x)