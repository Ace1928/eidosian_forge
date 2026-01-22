import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_index_private_image_visible_members_admin(self):
    request = unit_test_utils.get_fake_request(is_admin=True)
    output = self.controller.index(request, UUID4)
    self.assertEqual(1, len(output['members']))
    actual = set([image_member.member_id for image_member in output['members']])
    expected = set([TENANT1])
    self.assertEqual(expected, actual)