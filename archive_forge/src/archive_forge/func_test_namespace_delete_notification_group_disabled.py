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
def test_namespace_delete_notification_group_disabled(self):
    self.config(disabled_notifications=['metadef_namespace'])
    request = unit_test_utils.get_fake_request(tenant=TENANT2, roles=['admin'])
    self.namespace_controller.delete(request, NAMESPACE2)
    self.assertNotificationsLog([])
    self.assertRaises(webob.exc.HTTPNotFound, self.namespace_controller.show, request, NAMESPACE2)