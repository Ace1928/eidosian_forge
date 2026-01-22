import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_update_region_without_description_keeps_original(self):
    """Call ``PATCH /regions/{region_id}``."""
    region_ref = unit.new_region_ref()
    resp = self.post('/regions', body={'region': region_ref})
    region_updates = {'parent_region_id': self.region_id}
    resp = self.patch('/regions/%s' % region_ref['id'], body={'region': region_updates})
    self.assertEqual(region_ref['description'], resp.result['region']['description'])