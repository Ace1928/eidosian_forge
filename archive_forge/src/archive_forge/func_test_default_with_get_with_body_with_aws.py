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
def test_default_with_get_with_body_with_aws(self):
    request = wsgi.Request.blank('/?ContentType=JSON')
    request.method = 'GET'
    request.body = b'{"key": "value"}'
    actual = wsgi.JSONRequestDeserializer().default(request)
    expected = {'body': {'key': 'value'}}
    self.assertEqual(expected, actual)