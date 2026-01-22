import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_delete_private_image(self):
    enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'delete_member': "'{0}':%(owner)s".format(TENANT1), 'get_member': ''})
    self.controller.policy = enforcer
    request = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, UUID4, TENANT1)