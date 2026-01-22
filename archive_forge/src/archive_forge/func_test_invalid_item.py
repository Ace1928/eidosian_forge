from heat.common import identifier
from heat.tests import common
def test_invalid_item(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p')
    hi.identity['foo'] = 'bar'
    self.assertRaises(KeyError, lambda o, k: o[k], hi, 'foo')