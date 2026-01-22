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
def test_object_create_non_visible_namespace_admin(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT2, roles=['admin'])
    object = objects.MetadefObject()
    object.name = OBJECT2
    object.required = []
    object.properties = {}
    object = self.object_controller.create(request, object, NAMESPACE1)
    self.assertEqual(OBJECT2, object.name)
    self.assertEqual([], object.required)
    self.assertEqual({}, object.properties)
    self.assertNotificationLog('metadef_object.create', [{'name': OBJECT2, 'namespace': NAMESPACE1}])
    object = self.object_controller.show(request, NAMESPACE1, OBJECT2)
    self.assertEqual(OBJECT2, object.name)
    self.assertEqual([], object.required)
    self.assertEqual({}, object.properties)