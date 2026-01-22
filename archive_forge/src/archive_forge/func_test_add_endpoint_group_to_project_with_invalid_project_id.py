import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_add_endpoint_group_to_project_with_invalid_project_id(self):
    """Create an invalid endpoint group and project association."""
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    project_id = uuid.uuid4().hex
    url = self._get_project_endpoint_group_url(endpoint_group_id, project_id)
    self.put(url, expected_status=http.client.NOT_FOUND)