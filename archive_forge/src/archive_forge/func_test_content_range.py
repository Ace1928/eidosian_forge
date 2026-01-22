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
def test_content_range(self):
    request = wsgi.Request.blank('/tests/123')
    request.headers['Content-Range'] = 'bytes 10-99/*'
    range_ = request.get_range_from_request(120)
    self.assertEqual(10, range_.start)
    self.assertEqual(100, range_.stop)
    self.assertIsNone(range_.length)