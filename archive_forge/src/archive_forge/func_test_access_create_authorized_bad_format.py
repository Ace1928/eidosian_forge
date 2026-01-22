import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_access_create_authorized_bad_format(self):
    """Allow access, fail with 400"""
    rules = {'tasks_api_access': True, 'add_task': True}
    self.policy.set_rules(rules)
    request = unit_test_utils.get_fake_request()
    request.body = jsonutils.dump_as_bytes(self.bad_task)
    self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)