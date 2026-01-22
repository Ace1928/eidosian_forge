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
def test_object_update_conflict(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT3, roles=['admin'])
    object = self.object_controller.show(request, NAMESPACE3, OBJECT1)
    object.name = OBJECT2
    self.assertRaises(webob.exc.HTTPConflict, self.object_controller.update, request, object, NAMESPACE3, OBJECT1)
    self.assertNotificationsLog([])