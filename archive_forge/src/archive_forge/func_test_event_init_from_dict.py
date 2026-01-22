from heat.common import identifier
from heat.tests import common
def test_event_init_from_dict(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', '/resources/p/events/42')
    ei = identifier.EventIdentifier(**hi)
    self.assertEqual(hi, ei)