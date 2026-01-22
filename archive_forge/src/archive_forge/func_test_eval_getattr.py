import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_eval_getattr(self):
    m = ConcreteModel()
    m.T = RangeSet(3)

    @m.Block(m.T)
    def b(b, i):
        b.x = Var(initialize=i)

        @b.Block(m.T)
        def bb(bb, j):
            bb.I = RangeSet(i * j)
            bb.y = Var(bb.I, initialize=lambda m, i: i)
    t = IndexTemplate(m.T)
    e = m.b[t].x
    with self.assertRaisesRegex(ValueError, 'Evaluating uninitialized IndexTemplate \\({T}\\)'):
        value(e())
    with self.assertRaisesRegex(KeyError, "Index 'None' is not valid for indexed component 'b'"):
        self.assertIsNone(e(exception=False))
    with self.assertRaisesRegex(KeyError, "Index 'None' is not valid for indexed component 'b'"):
        self.assertIsNone(e(False))
    t.set_value(2)
    self.assertEqual(e(), 2)
    f = e.set_value(5)
    self.assertIs(f.__class__, CallExpression)
    self.assertEqual(f._kwds, ())
    self.assertEqual(len(f._args_), 2)
    self.assertIs(f._args_[0].__class__, EXPR.Structural_GetAttrExpression)
    self.assertIs(f._args_[0]._args_[0], e)
    self.assertEqual(f._args_[1], 5)
    self.assertEqual(value(m.b[2].x), 2)
    f()
    self.assertEqual(value(m.b[2].x), 5)
    f = e.set_value('a', skip_validation=True)
    self.assertIs(f.__class__, CallExpression)
    self.assertEqual(f._kwds, ('skip_validation',))
    self.assertEqual(len(f._args_), 3)
    self.assertIs(f._args_[0].__class__, EXPR.Structural_GetAttrExpression)
    self.assertIs(f._args_[0]._args_[0], e)
    self.assertEqual(f._args_[1], 'a')
    self.assertEqual(f._args_[2], True)
    f()
    self.assertEqual(value(m.b[2].x), 'a')