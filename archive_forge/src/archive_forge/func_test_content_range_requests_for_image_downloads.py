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
def test_content_range_requests_for_image_downloads(self):
    """
        Even though Content-Range is incorrect on requests, we support it
        for backward compatibility with clients written for pre-Pike
        Glance.
        The following test is for 'Content-Range' requests, which we have
        to ensure that we prevent regression.
        """

    def download_successful_ContentRange(d_range):
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Content-Range'] = d_range
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'X', b'Y', b'Z'])
        self.serializer.download(response, image)
        self.assertEqual(206, response.status_code)
        self.assertEqual('2', response.headers['Content-Length'])
        self.assertEqual('bytes 1-2/3', response.headers['Content-Range'])
        self.assertEqual(b'YZ', response.body)
    download_successful_ContentRange('bytes 1-2/3')
    download_successful_ContentRange('bytes 1-2/*')

    def download_failures_ContentRange(d_range):
        request = wsgi.Request.blank('/')
        request.environ = {}
        request.headers['Content-Range'] = d_range
        response = webob.Response()
        response.request = request
        image = FakeImage(size=3, data=[b'Z', b'Z', b'Z'])
        self.assertRaises(webob.exc.HTTPRequestRangeNotSatisfiable, self.serializer.download, response, image)
        return
    download_failures_ContentRange('bytes -3/3')
    download_failures_ContentRange('bytes 1-/3')
    download_failures_ContentRange('bytes 1-3/3')
    download_failures_ContentRange('bytes 1-4/3')
    download_failures_ContentRange('bytes 1-4/*')
    download_failures_ContentRange('bytes 4-1/3')
    download_failures_ContentRange('bytes 4-1/*')
    download_failures_ContentRange('bytes 4-8/*')
    download_failures_ContentRange('bytes 4-8/10')
    download_failures_ContentRange('bytes 4-8/3')