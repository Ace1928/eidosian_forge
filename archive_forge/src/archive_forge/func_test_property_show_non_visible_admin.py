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
def test_property_show_non_visible_admin(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT2, roles=['admin'])
    output = self.property_controller.show(request, NAMESPACE1, PROPERTY1)
    self.assertEqual(PROPERTY1, output.name)