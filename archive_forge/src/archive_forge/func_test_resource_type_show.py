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
def test_resource_type_show(self):
    request = unit_test_utils.get_fake_request()
    output = self.rt_controller.show(request, NAMESPACE3)
    self.assertEqual(1, len(output.resource_type_associations))
    actual = set([rt.name for rt in output.resource_type_associations])
    expected = set([RESOURCE_TYPE1])
    self.assertEqual(expected, actual)