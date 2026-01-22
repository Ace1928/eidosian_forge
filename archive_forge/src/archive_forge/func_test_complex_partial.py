from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_complex_partial(self):
    d = dependencies.Dependencies([('last', 'e1'), ('last', 'mid1'), ('last', 'mid2'), ('mid1', 'e2'), ('mid1', 'mid3'), ('mid2', 'mid3'), ('mid3', 'e3')])
    p = d['mid3']
    order = list(iter(p))
    self.assertEqual(4, len(order))
    for n in ('last', 'mid1', 'mid2', 'mid3'):
        self.assertIn(n, order, "'%s' not found in dependency order" % n)