import io
import pyomo.common.unittest as unittest
from pyomo.contrib.latex_printer import latex_printer
import pyomo.environ as pyo
from textwrap import dedent
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections.component_map import ComponentMap
from pyomo.environ import (
def generate_simple_model():
    import pyomo.environ as pyo
    m = pyo.ConcreteModel(name='basicFormulation')
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.objective_1 = pyo.Objective(expr=m.x + m.y)
    m.constraint_1 = pyo.Constraint(expr=m.x ** 2 + m.y ** 2.0 <= 1.0)
    m.constraint_2 = pyo.Constraint(expr=m.x >= 0.0)
    m.I = pyo.Set(initialize=[1, 2, 3, 4, 5])
    m.J = pyo.Set(initialize=[1, 2, 3])
    m.K = pyo.Set(initialize=[1, 3, 5])
    m.u = pyo.Var(m.I * m.I)
    m.v = pyo.Var(m.I)
    m.w = pyo.Var(m.J)
    m.p = pyo.Var(m.K)

    def ruleMaker(m, j):
        return (m.x + m.y) * sum((m.v[i] + m.u[i, j] ** 2 for i in m.I)) <= 0
    m.constraint_7 = pyo.Constraint(m.I, rule=ruleMaker)

    def ruleMaker(m):
        return sum((m.p[k] for k in m.K)) == 1
    m.constraint_8 = pyo.Constraint(rule=ruleMaker)
    return m