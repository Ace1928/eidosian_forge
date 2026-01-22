import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_show_by_member(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT4)
    output = self.controller.show(request, UUID2, TENANT4)
    expected = self.image_members[0]
    self.assertEqual(expected['image_id'], output.image_id)
    self.assertEqual(expected['member'], output.member_id)
    self.assertEqual(expected['status'], output.status)