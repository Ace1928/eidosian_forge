import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_endpoint_group(self):
    """POST /OS-EP-FILTER/endpoint_groups.

        Valid endpoint group test case.

        """
    r = self.post(self.DEFAULT_ENDPOINT_GROUP_URL, body=self.DEFAULT_ENDPOINT_GROUP_BODY)
    expected_filters = self.DEFAULT_ENDPOINT_GROUP_BODY['endpoint_group']['filters']
    expected_name = self.DEFAULT_ENDPOINT_GROUP_BODY['endpoint_group']['name']
    self.assertEqual(expected_filters, r.result['endpoint_group']['filters'])
    self.assertEqual(expected_name, r.result['endpoint_group']['name'])
    self.assertThat(r.result['endpoint_group']['links']['self'], matchers.EndsWith('/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': r.result['endpoint_group']['id']}))