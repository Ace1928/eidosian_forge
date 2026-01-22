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
def test_resource_type_association_delete_disabled_notification(self):
    self.config(disabled_notifications=['metadef_resource_type.delete'])
    request = unit_test_utils.get_fake_request(tenant=TENANT3, roles=['admin'])
    self.rt_controller.delete(request, NAMESPACE3, RESOURCE_TYPE1)
    self.assertNotificationsLog([])
    output = self.rt_controller.show(request, NAMESPACE3)
    self.assertEqual(0, len(output.resource_type_associations))