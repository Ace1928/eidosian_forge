import pyomo.common.unittest as unittest
from pyomo.opt import (
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
@unittest.skipIf(msk_version[0] > 9, 'MOSEK 10 does not (yet) have a removeacc method.')
def test_constraint_removal_2(self):
    m = pmo.block()
    m.x = pmo.variable()
    m.y = pmo.variable()
    m.z = pmo.variable()
    m.c1 = pmo.conic.rotated_quadratic.as_domain(2, m.x, [m.y])
    m.c2 = pmo.conic.quadratic(m.x, [m.y, m.z])
    m.c3 = pmo.constraint(m.z >= 0)
    m.c4 = pmo.constraint(m.x + m.y >= 0)
    opt = pmo.SolverFactory('mosek_persistent')
    opt.set_instance(m)
    self.assertEqual(opt._solver_model.getnumcon(), 5)
    self.assertEqual(opt._solver_model.getnumcone(), 2)
    opt.remove_block(m.c1)
    self.assertEqual(opt._solver_model.getnumcon(), 2)
    self.assertEqual(opt._solver_model.getnumcone(), 1)
    opt.remove_constraints(m.c2, m.c3)
    self.assertEqual(opt._solver_model.getnumcon(), 1)
    self.assertEqual(opt._solver_model.getnumcone(), 0)
    self.assertRaises(ValueError, opt.remove_constraint, m.c2)
    opt.add_constraint(m.c2)
    opt.add_block(m.c1)
    self.assertEqual(opt._solver_model.getnumcone(), 2)