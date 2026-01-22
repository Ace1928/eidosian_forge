import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_endpoint_group_in_project(self):
    """Test retrieving project endpoint group association."""
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    url = self._get_project_endpoint_group_url(endpoint_group_id, self.project_id)
    self.put(url)
    response = self.get(url)
    self.assertEqual(endpoint_group_id, response.result['project_endpoint_group']['endpoint_group_id'])
    self.assertEqual(self.project_id, response.result['project_endpoint_group']['project_id'])