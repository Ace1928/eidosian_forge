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
def test_namespace_update_name(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    namespace = self.namespace_controller.show(request, NAMESPACE1)
    namespace.namespace = NAMESPACE4
    namespace = self.namespace_controller.update(request, namespace, NAMESPACE1)
    self.assertEqual(NAMESPACE4, namespace.namespace)
    self.assertNotificationLog('metadef_namespace.update', [{'namespace': NAMESPACE4, 'namespace_old': NAMESPACE1}])
    namespace = self.namespace_controller.show(request, NAMESPACE4)
    self.assertEqual(NAMESPACE4, namespace.namespace)
    self.assertRaises(webob.exc.HTTPNotFound, self.namespace_controller.show, request, NAMESPACE1)