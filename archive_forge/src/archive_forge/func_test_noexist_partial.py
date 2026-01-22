from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_noexist_partial(self):
    d = dependencies.Dependencies([('foo', 'bar')])

    def get(i):
        return d[i]
    self.assertRaises(KeyError, get, 'baz')