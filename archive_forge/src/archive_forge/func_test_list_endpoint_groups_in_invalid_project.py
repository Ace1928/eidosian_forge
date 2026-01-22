import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_endpoint_groups_in_invalid_project(self):
    """Test retrieving from invalid project."""
    project_id = uuid.uuid4().hex
    url = '/OS-EP-FILTER/projects/%(project_id)s/endpoint_groups' % {'project_id': project_id}
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)