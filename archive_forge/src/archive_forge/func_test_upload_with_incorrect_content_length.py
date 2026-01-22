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
def test_upload_with_incorrect_content_length(self):
    request = unit_test_utils.get_fake_request()
    request.headers['Content-Type'] = 'application/octet-stream'
    request.body = b'YYY'
    request.headers['Content-Length'] = 4
    output = self.deserializer.upload(request)
    data = output.pop('data')
    self.assertEqual(b'YYY', data.read())
    expected = {'size': 4}
    self.assertEqual(expected, output)