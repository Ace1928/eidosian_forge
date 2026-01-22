import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
def check_pw_paraboloid(self, m):
    z = m.pw_paraboloid.get_transformation_var(m.paraboloid_expr)
    self.assertIsInstance(z, Var)
    paraboloid_block = z.parent_block()
    ct.check_trans_block_structure(self, paraboloid_block)
    self.assertEqual(len(paraboloid_block.disjuncts), 4)
    disjuncts_dict = {paraboloid_block.disjuncts[0]: ([(0, 1), (0, 4), (3, 4)], m.g1), paraboloid_block.disjuncts[1]: ([(0, 1), (3, 4), (3, 1)], m.g1), paraboloid_block.disjuncts[2]: ([(3, 4), (3, 7), (0, 7)], m.g2), paraboloid_block.disjuncts[3]: ([(0, 7), (0, 4), (3, 4)], m.g2)}
    for d, (pts, f) in disjuncts_dict.items():
        self.check_paraboloid_disjunct(d, pts, f, paraboloid_block.substitute_var, m.x1, m.x2)
    self.assertIsInstance(paraboloid_block.pick_a_piece, Disjunction)
    self.assertEqual(len(paraboloid_block.pick_a_piece.disjuncts), 4)
    for i in range(3):
        self.assertIs(paraboloid_block.pick_a_piece.disjuncts[i], paraboloid_block.disjuncts[i])
    self.assertIs(m.indexed_c[0].body.args[0].expr, paraboloid_block.substitute_var)