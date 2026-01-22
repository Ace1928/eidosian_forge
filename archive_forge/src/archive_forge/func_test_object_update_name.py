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
def test_object_update_name(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    object = self.object_controller.show(request, NAMESPACE1, OBJECT1)
    object.name = OBJECT2
    object = self.object_controller.update(request, object, NAMESPACE1, OBJECT1)
    self.assertEqual(OBJECT2, object.name)
    self.assertNotificationLog('metadef_object.update', [{'name': OBJECT2, 'name_old': OBJECT1, 'namespace': NAMESPACE1}])
    object = self.object_controller.show(request, NAMESPACE1, OBJECT2)
    self.assertEqual(OBJECT2, object.name)