from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_simple_partial(self):
    d = dependencies.Dependencies([('last', 'middle'), ('middle', 'first')])
    p = d['middle']
    order = list(iter(p))
    self.assertEqual(2, len(order))
    for n in ('last', 'middle'):
        self.assertIn(n, order, "'%s' not found in dependency order" % n)
    self.assertGreater(order.index('last'), order.index('middle'))