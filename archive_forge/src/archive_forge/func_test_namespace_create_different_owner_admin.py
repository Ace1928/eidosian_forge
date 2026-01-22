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
def test_namespace_create_different_owner_admin(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    namespace = namespaces.Namespace()
    namespace.namespace = NAMESPACE4
    namespace.owner = TENANT4
    namespace = self.namespace_controller.create(request, namespace)
    self.assertEqual(NAMESPACE4, namespace.namespace)
    self.assertNotificationLog('metadef_namespace.create', [{'namespace': NAMESPACE4}])
    namespace = self.namespace_controller.show(request, NAMESPACE4)
    self.assertEqual(NAMESPACE4, namespace.namespace)