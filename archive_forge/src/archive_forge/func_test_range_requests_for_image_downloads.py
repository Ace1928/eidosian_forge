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
def test_range_requests_for_image_downloads(self):
    """
        Test partial download 'Range' requests for images (random image access)
        """

    def download_successful_Range(d_range):
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Range'] = d_range
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'X', b'Y', b'Z'])
        self.serializer.download(response, image)
        self.assertEqual(206, response.status_code)
        self.assertEqual('2', response.headers['Content-Length'])
        self.assertEqual('bytes 1-2/3', response.headers['Content-Range'])
        self.assertEqual(b'YZ', response.body)
    download_successful_Range('bytes=1-2')
    download_successful_Range('bytes=1-')
    download_successful_Range('bytes=1-3')
    download_successful_Range('bytes=-2')
    download_successful_Range('bytes=1-100')

    def full_image_download_w_range(d_range):
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Range'] = d_range
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'X', b'Y', b'Z'])
        self.serializer.download(response, image)
        self.assertEqual(206, response.status_code)
        self.assertEqual('3', response.headers['Content-Length'])
        self.assertEqual('bytes 0-2/3', response.headers['Content-Range'])
        self.assertEqual(b'XYZ', response.body)
    full_image_download_w_range('bytes=0-')
    full_image_download_w_range('bytes=0-2')
    full_image_download_w_range('bytes=0-3')
    full_image_download_w_range('bytes=-3')
    full_image_download_w_range('bytes=-4')
    full_image_download_w_range('bytes=0-100')
    full_image_download_w_range('bytes=-100')

    def download_failures_Range(d_range):
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Range'] = d_range
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
        self.assertRaises(webob.exc.HTTPRequestRangeNotSatisfiable, self.serializer.download, response, image)
        return
    download_failures_Range('bytes=4-1')
    download_failures_Range('bytes=4-')
    download_failures_Range('bytes=3-')
    download_failures_Range('bytes=1')
    download_failures_Range('bytes=100')
    download_failures_Range('bytes=100-')
    download_failures_Range('bytes=')