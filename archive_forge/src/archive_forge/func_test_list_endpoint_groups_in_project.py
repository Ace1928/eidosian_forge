import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_endpoint_groups_in_project(self):
    """GET & HEAD /OS-EP-FILTER/projects/{project_id}/endpoint_groups."""
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    url = self._get_project_endpoint_group_url(endpoint_group_id, self.project_id)
    self.put(url)
    url = '/OS-EP-FILTER/projects/%(project_id)s/endpoint_groups' % {'project_id': self.project_id}
    response = self.get(url, expected_status=http.client.OK)
    self.assertEqual(endpoint_group_id, response.result['endpoint_groups'][0]['id'])
    self.head(url, expected_status=http.client.OK)