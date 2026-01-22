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
def test_has_body_has_wrong_content_type(self):
    request = wsgi.Request.blank('/')
    request.method = 'POST'
    request.body = b'{"key": "value"}'
    self.assertIn('Content-Length', request.headers)
    request.headers['Content-Type'] = 'application/xml'
    self.assertFalse(wsgi.JSONRequestDeserializer().has_body(request))