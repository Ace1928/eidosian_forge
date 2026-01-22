import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_remove_endpoint_group_with_project_association(self):
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    project_endpoint_group_url = self._get_project_endpoint_group_url(endpoint_group_id, self.default_domain_project_id)
    self.put(project_endpoint_group_url)
    endpoint_group_url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    self.delete(endpoint_group_url)
    self.get(endpoint_group_url, expected_status=http.client.NOT_FOUND)
    self.get(project_endpoint_group_url, expected_status=http.client.NOT_FOUND)