import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_region_with_conflicting_ids(self):
    """Call ``PUT /regions/{region_id}`` with conflicting region IDs."""
    ref = unit.new_region_ref()
    self.put('/regions/%s' % uuid.uuid4().hex, body={'region': ref}, expected_status=http.client.BAD_REQUEST)