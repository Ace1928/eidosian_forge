import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_get_head_service(self):
    """Call ``GET & HEAD /services/{service_id}``."""
    resource_url = '/services/%(service_id)s' % {'service_id': self.service_id}
    r = self.get(resource_url)
    self.assertValidServiceResponse(r, self.service)
    self.head(resource_url, expected_status=http.client.OK)