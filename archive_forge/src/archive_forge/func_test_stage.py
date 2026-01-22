import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_stage(self):
    request = webob.Request.blank('/')
    request.environ = {}
    response = webob.Response()
    response.request = request
    self.serializer.stage(response, {})
    self.assertEqual(http.NO_CONTENT, response.status_int)
    self.assertEqual('0', response.headers['Content-Length'])