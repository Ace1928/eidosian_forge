import uuid
from osc_placement.tests.functional import base
def test_allocation_delete_not_found(self):
    consumer_uuid = str(uuid.uuid4())
    msg = "No allocations for consumer '{}'".format(consumer_uuid)
    exc = self.assertRaises(base.CommandException, self.resource_allocation_delete, consumer_uuid)
    self.assertIn(msg, str(exc))