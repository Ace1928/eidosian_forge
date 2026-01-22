import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_endpoints_associated_with_invalid_project(self):
    """GET & HEAD /OS-EP-FILTER/projects/{project_id}/endpoints.

        Invalid project id test case.

        """
    self.put(self.default_request_url)
    url = '/OS-EP-FILTER/projects/%(project_id)s/endpoints' % {'project_id': uuid.uuid4().hex}
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)