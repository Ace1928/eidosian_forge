import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_endpoints_associated_with_endpoint_group(self):
    """GET & HEAD /OS-EP-FILTER/endpoint_groups/{endpoint_group}/endpoints.

        Valid endpoint group test case.

        """
    service_ref = unit.new_service_ref()
    response = self.post('/services', body={'service': service_ref})
    service_id = response.result['service']['id']
    endpoint_ref = unit.new_endpoint_ref(service_id=service_id, interface='public', region_id=self.region_id)
    response = self.post('/endpoints', body={'endpoint': endpoint_ref})
    endpoint_id = response.result['endpoint']['id']
    body = copy.deepcopy(self.DEFAULT_ENDPOINT_GROUP_BODY)
    body['endpoint_group']['filters'] = {'service_id': service_id}
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, body)
    self._create_endpoint_group_project_association(endpoint_group_id, self.project_id)
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s/endpoints' % {'endpoint_group_id': endpoint_group_id}
    r = self.get(url, expected_status=http.client.OK)
    self.assertNotEmpty(r.result['endpoints'])
    self.assertEqual(endpoint_id, r.result['endpoints'][0].get('id'))
    self.head(url, expected_status=http.client.OK)