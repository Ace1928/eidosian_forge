from pyomo.environ import ConcreteModel, Var, ExternalFunction, Objective
from pyomo.opt import SolverFactory
def basis_rule(component, ef_expr):
    x = ef_expr.arg(0)
    y = ef_expr.arg(1)
    return x ** 2 - y