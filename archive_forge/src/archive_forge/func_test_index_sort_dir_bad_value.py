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
def test_index_sort_dir_bad_value(self):
    request = unit_test_utils.get_fake_request('/tasks?sort_dir=invalid')
    self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)