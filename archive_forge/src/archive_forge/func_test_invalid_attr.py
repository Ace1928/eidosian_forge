from heat.common import identifier
from heat.tests import common
def test_invalid_attr(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p')
    hi.identity['foo'] = 'bar'
    self.assertRaises(AttributeError, getattr, hi, 'foo')