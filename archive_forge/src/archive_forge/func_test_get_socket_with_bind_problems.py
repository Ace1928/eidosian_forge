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
def test_get_socket_with_bind_problems(self):
    self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.eventlet.listen', mock.Mock(side_effect=[wsgi.socket.error(socket.errno.EADDRINUSE)] * 3 + [None])))
    self.assertRaises(RuntimeError, wsgi.get_socket, 1234)