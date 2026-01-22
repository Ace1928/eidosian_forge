import uuid
from osc_placement.tests.functional import base
def test_invalid_version(self):
    """Negative test to ensure the unset command requires >= 1.12."""
    consumer_uuid = str(uuid.uuid4())
    self.assertCommandFailed('requires at least version 1.12', self.resource_allocation_unset, consumer_uuid)