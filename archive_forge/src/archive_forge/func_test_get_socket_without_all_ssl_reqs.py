from unittest import mock
import fixtures
import json
from oslo_config import cfg
import socket
import webob
from heat.api.aws import exception as aws_exception
from heat.common import exception
from heat.common import wsgi
from heat.tests import common
def test_get_socket_without_all_ssl_reqs(self):
    wsgi.cfg.CONF.heat_api.key_file = None
    self.assertRaises(RuntimeError, wsgi.get_socket, wsgi.cfg.CONF.heat_api, 1234)