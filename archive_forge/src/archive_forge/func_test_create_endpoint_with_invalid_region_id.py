import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_endpoint_with_invalid_region_id(self):
    """Call ``POST /endpoints``."""
    ref = unit.new_endpoint_ref(service_id=self.service_id)
    self.post('/endpoints', body={'endpoint': ref}, expected_status=http.client.BAD_REQUEST)