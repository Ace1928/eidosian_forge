import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_get_head_region(self):
    """Call ``GET & HEAD /regions/{region_id}``."""
    resource_url = '/regions/%(region_id)s' % {'region_id': self.region_id}
    r = self.get(resource_url)
    self.assertValidRegionResponse(r, self.region)
    self.head(resource_url, expected_status=http.client.OK)