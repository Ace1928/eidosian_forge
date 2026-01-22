from heat.common import identifier
from heat.tests import common
def test_event_stack(self):
    si = identifier.HeatIdentifier('t', 's', 'i')
    pi = identifier.ResourceIdentifier(resource_name='r', **si)
    ei = identifier.EventIdentifier(event_id='e', **pi)
    self.assertEqual(si, ei.stack())