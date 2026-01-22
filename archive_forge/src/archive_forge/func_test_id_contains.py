from heat.common import identifier
from heat.tests import common
def test_id_contains(self):
    hi = identifier.HeatIdentifier('t', 's', ':/')
    self.assertNotIn('t', hi)
    self.assertIn('stack_id', hi)