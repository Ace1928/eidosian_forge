import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_endpoint_group_project_cleanup_with_project(self):
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    project_ref = unit.new_project_ref(domain_id=self.domain_id)
    r = self.post('/projects', body={'project': project_ref})
    project = self.assertValidProjectResponse(r, project_ref)
    url = self._get_project_endpoint_group_url(endpoint_group_id, project['id'])
    self.put(url)
    self.get(url, expected_status=http.client.OK)
    self.get(url, expected_status=http.client.OK)
    self.delete('/projects/%(project_id)s' % {'project_id': project['id']})
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)