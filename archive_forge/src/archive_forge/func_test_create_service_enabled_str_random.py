import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_service_enabled_str_random(self):
    """Call ``POST /services``."""
    ref = unit.new_service_ref(enabled='puppies')
    self.post('/services', body={'service': ref}, expected_status=http.client.BAD_REQUEST)