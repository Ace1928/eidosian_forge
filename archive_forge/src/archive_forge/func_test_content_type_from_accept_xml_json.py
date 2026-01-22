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
def test_content_type_from_accept_xml_json(self):
    request = wsgi.Request.blank('/tests/123')
    request.headers['Accept'] = 'application/xml, application/json'
    result = request.best_match_content_type()
    self.assertEqual('application/json', result)