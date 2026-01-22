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
def test_resource_type_association_create_non_existing_namespace(self):
    request = unit_test_utils.get_fake_request(roles=['admin'])
    rt = resource_types.ResourceTypeAssociation()
    rt.name = RESOURCE_TYPE1
    rt.prefix = 'pref'
    self.assertRaises(webob.exc.HTTPNotFound, self.rt_controller.create, request, rt, NAMESPACE4)
    self.assertNotificationsLog([])