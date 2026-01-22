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
def test_namespace_create_with_related_resources(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    namespace = namespaces.Namespace()
    namespace.namespace = NAMESPACE4
    prop1 = properties.PropertyType()
    prop1.type = 'string'
    prop1.title = 'title'
    prop2 = properties.PropertyType()
    prop2.type = 'string'
    prop2.title = 'title'
    namespace.properties = {PROPERTY1: prop1, PROPERTY2: prop2}
    object1 = objects.MetadefObject()
    object1.name = OBJECT1
    object1.required = []
    object1.properties = {}
    object2 = objects.MetadefObject()
    object2.name = OBJECT2
    object2.required = []
    object2.properties = {}
    namespace.objects = [object1, object2]
    output = self.namespace_controller.create(request, namespace)
    self.assertEqual(NAMESPACE4, namespace.namespace)
    output = output.to_dict()
    self.assertEqual(2, len(output['properties']))
    actual = set([property for property in output['properties']])
    expected = set([PROPERTY1, PROPERTY2])
    self.assertEqual(expected, actual)
    self.assertEqual(2, len(output['objects']))
    actual = set([object.name for object in output['objects']])
    expected = set([OBJECT1, OBJECT2])
    self.assertEqual(expected, actual)
    output = self.namespace_controller.show(request, NAMESPACE4)
    self.assertEqual(NAMESPACE4, namespace.namespace)
    output = output.to_dict()
    self.assertEqual(2, len(output['properties']))
    actual = set([property for property in output['properties']])
    expected = set([PROPERTY1, PROPERTY2])
    self.assertEqual(expected, actual)
    self.assertEqual(2, len(output['objects']))
    actual = set([object.name for object in output['objects']])
    expected = set([OBJECT1, OBJECT2])
    self.assertEqual(expected, actual)
    self.assertNotificationsLog([{'type': 'metadef_namespace.create', 'payload': {'namespace': NAMESPACE4, 'owner': TENANT1}}, {'type': 'metadef_object.create', 'payload': {'namespace': NAMESPACE4, 'name': OBJECT1, 'properties': []}}, {'type': 'metadef_object.create', 'payload': {'namespace': NAMESPACE4, 'name': OBJECT2, 'properties': []}}, {'type': 'metadef_property.create', 'payload': {'namespace': NAMESPACE4, 'type': 'string', 'title': 'title'}}, {'type': 'metadef_property.create', 'payload': {'namespace': NAMESPACE4, 'type': 'string', 'title': 'title'}}])