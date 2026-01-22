import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_endpoint_group(self):
    """GET /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Valid endpoint group test case.

        """
    response = self.post(self.DEFAULT_ENDPOINT_GROUP_URL, body=self.DEFAULT_ENDPOINT_GROUP_BODY)
    endpoint_group_id = response.result['endpoint_group']['id']
    endpoint_group_filters = response.result['endpoint_group']['filters']
    endpoint_group_name = response.result['endpoint_group']['name']
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    self.get(url)
    self.assertEqual(endpoint_group_id, response.result['endpoint_group']['id'])
    self.assertEqual(endpoint_group_filters, response.result['endpoint_group']['filters'])
    self.assertEqual(endpoint_group_name, response.result['endpoint_group']['name'])
    self.assertThat(response.result['endpoint_group']['links']['self'], matchers.EndsWith(url))