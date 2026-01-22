import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_update_done_by_owner(self):
    enforcer = unit_test_utils.enforcer_from_rules({'get_image': '', 'modify_image': "'{0}':%(owner)s".format(TENANT1)})
    self.controller.policy = enforcer
    request = unit_test_utils.get_fake_request(tenant=TENANT1)
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, UUID2, TENANT4, status='accepted')