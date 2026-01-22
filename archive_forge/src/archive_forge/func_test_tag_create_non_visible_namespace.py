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
def test_tag_create_non_visible_namespace(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT2)
    self.assertRaises(webob.exc.HTTPNotFound, self.tag_controller.create, request, NAMESPACE1, TAG1)
    self.assertNotificationsLog([])