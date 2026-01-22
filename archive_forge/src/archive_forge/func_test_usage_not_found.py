import operator
import uuid
from osc_placement.tests.functional import base
def test_usage_not_found(self):
    rp_uuid = str(uuid.uuid4())
    exc = self.assertRaises(base.CommandException, self.resource_provider_show_usage, rp_uuid)
    self.assertIn('No resource provider with uuid {} found'.format(rp_uuid), str(exc))