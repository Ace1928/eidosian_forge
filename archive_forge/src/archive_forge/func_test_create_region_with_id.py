import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_region_with_id(self):
    """Call ``PUT /regions/{region_id}`` w/o an ID in the request body."""
    ref = unit.new_region_ref()
    region_id = ref.pop('id')
    r = self.put('/regions/%s' % region_id, body={'region': ref}, expected_status=http.client.CREATED)
    self.assertValidRegionResponse(r, ref)
    self.assertEqual(region_id, r.json['region']['id'])