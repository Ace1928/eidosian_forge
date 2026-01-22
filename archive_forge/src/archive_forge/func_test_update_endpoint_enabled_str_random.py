import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_update_endpoint_enabled_str_random(self):
    """Call ``PATCH /endpoints/{endpoint_id}`` with enabled: 'kitties'."""
    self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': {'enabled': 'kitties'}}, expected_status=http.client.BAD_REQUEST)