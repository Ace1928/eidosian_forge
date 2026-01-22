import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_create_unlimited(self):
    self.config(image_member_quota=-1)
    request = unit_test_utils.get_fake_request()
    image_id = UUID2
    member_id = TENANT3
    output = self.controller.create(request, image_id=image_id, member_id=member_id)
    self.assertEqual(UUID2, output.image_id)
    self.assertEqual(TENANT3, output.member_id)