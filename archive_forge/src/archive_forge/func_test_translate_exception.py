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
@mock.patch.object(i18n, 'translate')
def test_translate_exception(self, mock_translate, mock_lookup):
    mock_translate.return_value = 'No Encontrado'
    mock_lookup.return_value = 'de'
    req = wsgi.Request.blank('/tests/123')
    req.headers['Accept-Language'] = 'de'
    e = webob.exc.HTTPNotFound(explanation='Not Found')
    e = wsgi.translate_exception(req, e)
    self.assertEqual('No Encontrado', e.explanation)