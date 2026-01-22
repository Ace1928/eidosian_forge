from heat.common import identifier
from heat.tests import common
def test_equal_dict(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p')
    self.assertTrue(hi == dict(hi))
    self.assertTrue(dict(hi) == hi)