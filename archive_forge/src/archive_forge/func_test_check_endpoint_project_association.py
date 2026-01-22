import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_check_endpoint_project_association(self):
    """HEAD /OS-EP-FILTER/projects/{project_id}/endpoints/{endpoint_id}.

        Valid project and endpoint id test case.

        """
    self.put(self.default_request_url)
    self.head('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': self.endpoint_id}, expected_status=http.client.NO_CONTENT)