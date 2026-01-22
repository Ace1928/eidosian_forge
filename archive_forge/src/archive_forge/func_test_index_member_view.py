import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_index_member_view(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT4)
    output = self.controller.index(request, UUID3)
    self.assertEqual(1, len(output['members']))
    actual = set([image_member.member_id for image_member in output['members']])
    expected = set([TENANT4])
    self.assertEqual(expected, actual)