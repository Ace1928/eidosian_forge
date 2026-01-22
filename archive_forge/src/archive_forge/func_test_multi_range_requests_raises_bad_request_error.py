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
def test_multi_range_requests_raises_bad_request_error(self):
    request = wsgi.Request.blank('/')
    request.environ = {}
    request.headers['Range'] = 'bytes=0-0,-1'
    response = webob.Response()
    response.request = request
    image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
    self.assertRaises(webob.exc.HTTPBadRequest, self.serializer.download, response, image)