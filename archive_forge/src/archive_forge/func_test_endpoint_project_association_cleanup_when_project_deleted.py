import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_endpoint_project_association_cleanup_when_project_deleted(self):
    self.put(self.default_request_url)
    association_url = '/OS-EP-FILTER/endpoints/%(endpoint_id)s/projects' % {'endpoint_id': self.endpoint_id}
    r = self.get(association_url)
    self.assertValidProjectListResponse(r, expected_length=1)
    self.delete('/projects/%(project_id)s' % {'project_id': self.default_domain_project_id})
    r = self.get(association_url)
    self.assertValidProjectListResponse(r, expected_length=0)