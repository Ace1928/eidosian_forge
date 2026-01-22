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
def test_namespace_create_duplicate(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    namespace = namespaces.Namespace()
    namespace.namespace = 'new-namespace'
    new_ns = self.namespace_controller.create(request, namespace)
    self.assertEqual('new-namespace', new_ns.namespace)
    self.assertRaises(webob.exc.HTTPConflict, self.namespace_controller.create, request, namespace)