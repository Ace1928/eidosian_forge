import http.client as http
import webob
import glance.api.v2.image_tags
from glance.common import exception
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.unit.v2.test_image_data_resource as image_data_tests
import glance.tests.utils as test_utils
def test_create_duplicate_tag_ignored(self):
    request = unit_test_utils.get_fake_request()
    self.controller.update(request, unit_test_utils.UUID1, 'dink')
    self.controller.update(request, unit_test_utils.UUID1, 'dink')
    context = request.context
    tags = self.db.image_tag_get_all(context, unit_test_utils.UUID1)
    self.assertEqual(1, len([tag for tag in tags if tag == 'dink']))