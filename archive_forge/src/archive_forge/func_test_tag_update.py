import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_tag_update(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT3, roles=['admin'])
    tag = self.tag_controller.show(request, NAMESPACE3, TAG1)
    tag.name = TAG3
    tag = self.tag_controller.update(request, tag, NAMESPACE3, TAG1)
    self.assertEqual(TAG3, tag.name)
    self.assertNotificationLog('metadef_tag.update', [{'name': TAG3, 'namespace': NAMESPACE3}])
    property = self.tag_controller.show(request, NAMESPACE3, TAG3)
    self.assertEqual(TAG3, property.name)