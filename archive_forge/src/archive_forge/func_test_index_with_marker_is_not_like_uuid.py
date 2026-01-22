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
def test_index_with_marker_is_not_like_uuid(self):
    marker = 'INVALID_UUID'
    path = '/tasks'
    request = unit_test_utils.get_fake_request(path)
    self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, marker=marker)