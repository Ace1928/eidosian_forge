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
@mock.patch.object(webob.acceptparse.AcceptLanguageValidHeader, 'lookup')
def test_best_match_language_unknown(self, mock_lookup):
    request = wsgi.Request.blank('/')
    accepted = 'unknown-lang'
    request.headers = {'Accept-Language': accepted}
    mock_lookup.return_value = 'fake_LANG'
    self.assertIsNone(request.best_match_language())
    mock_lookup.assert_called_once()
    request.headers = {'Accept-Language': ''}
    self.assertIsNone(request.best_match_language())
    request.headers.pop('Accept-Language')
    self.assertIsNone(request.best_match_language())