import pyomo.common.unittest as unittest
from pyomo.opt import (
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
def test_column_addition(self):
    """
        Test based on lo1.py problem from MOSEK documentation.
        """
    sol_to_get = [0.0, 0.0, 15.0, 8]
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None))
    m.y = pyo.Var(bounds=(0, 10))
    m.z = pyo.Var(bounds=(0, None))
    m.c1 = pyo.Constraint(expr=3 * m.x + m.y + 2 * m.z == 30)
    m.c2 = pyo.Constraint(expr=2 * m.x + m.y + 3 * m.z >= 15)
    m.c3 = pyo.Constraint(expr=2 * m.y <= 25)
    m.o = pyo.Objective(expr=3 * m.x + m.y + 5 * m.z, sense=pyo.maximize)
    opt = pyo.SolverFactory('mosek_persistent')
    opt.set_instance(m)
    m.new_var = pyo.Var(bounds=(0, None))
    opt.add_column(m, m.new_var, 1, [m.c2, m.c3], [1, 3])
    self.assertEqual(opt._solver_model.getnumvar(), 4)
    opt.solve(m)
    for i, v in enumerate([m.x, m.y, m.z, m.new_var]):
        with self.subTest(i=v.name):
            self.assertAlmostEqual(v.value, sol_to_get[i], places=0)