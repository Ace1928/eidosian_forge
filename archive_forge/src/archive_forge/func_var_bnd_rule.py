from pyomo.environ import ConcreteModel, Var, Objective, Constraint, RangeSet
def var_bnd_rule(model, i):
    return (-1.0, model.x[1, i], 1.0)