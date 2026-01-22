import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
def test_call_raises_exception(self):

    class FakeController(object):

        def index(self, shirt, pants=None):
            return (shirt, pants)
    resource = wsgi.Resource(FakeController(), None, None)

    def dispatch(self, obj, action, *args, **kwargs):
        raise Exception('test exception')
    self.mock_object(wsgi.Resource, 'dispatch', dispatch)
    request = wsgi.Request.blank('/')
    response = resource.__call__(request)
    self.assertIsInstance(response, webob.exc.HTTPInternalServerError)
    self.assertEqual(http.INTERNAL_SERVER_ERROR, response.status_code)