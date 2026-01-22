import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_filter_list_services_by_type(self):
    """Call ``GET /services?type=<some type>``."""
    target_ref = self._create_random_service()
    self._create_random_service()
    self._create_random_service()
    response = self.get('/services?type=' + target_ref['type'])
    self.assertValidServiceListResponse(response, ref=target_ref)
    filtered_service_list = response.json['services']
    self.assertEqual(1, len(filtered_service_list))
    filtered_service = filtered_service_list[0]
    self.assertEqual(target_ref['type'], filtered_service['type'])