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
def test_resource_type_index(self):
    request = unit_test_utils.get_fake_request()
    output = self.rt_controller.index(request)
    self.assertEqual(3, len(output.resource_types))
    actual = set([rtype.name for rtype in output.resource_types])
    expected = set([RESOURCE_TYPE1, RESOURCE_TYPE2, RESOURCE_TYPE4])
    self.assertEqual(expected, actual)