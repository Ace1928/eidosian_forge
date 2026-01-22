import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_index_allowed_by_get_members_policy(self):
    rules = {'get_members': True}
    self.policy.set_rules(rules)
    request = unit_test_utils.get_fake_request()
    output = self.controller.index(request, UUID2)
    self.assertEqual(1, len(output['members']))