from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_single_partial(self):
    d = dependencies.Dependencies([('last', 'first')])
    p = d['last']
    li = list(iter(p))
    self.assertEqual(1, len(li))
    self.assertEqual('last', li[0])