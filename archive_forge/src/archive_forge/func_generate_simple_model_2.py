import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def generate_simple_model_2():
    import pyomo.environ as pyo
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x_dot = pyo.Var()
    m.x_bar = pyo.Var()
    m.x_star = pyo.Var()
    m.x_hat = pyo.Var()
    m.x_hat_1 = pyo.Var()
    m.y_sub1_sub2_sub3 = pyo.Var()
    m.objective_1 = pyo.Objective(expr=m.y_sub1_sub2_sub3)
    m.constraint_1 = pyo.Constraint(expr=(m.x_dot + m.x_bar + m.x_star + m.x_hat + m.x_hat_1) ** 2 <= m.y_sub1_sub2_sub3)
    m.constraint_2 = pyo.Constraint(expr=(m.x_dot + m.x_bar) ** (-(m.x_star + m.x_hat)) <= m.y_sub1_sub2_sub3)
    m.constraint_3 = pyo.Constraint(expr=-(m.x_dot + m.x_bar) + -(m.x_star + m.x_hat) <= m.y_sub1_sub2_sub3)
    return m