from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
def test_overwrites_auth_url_from_headers_with_local_config(self):
    req = webob.Request.blank('/tenant_id/')
    req.headers['X-Auth-Url'] = 'should_be_overwritten'
    self.middleware(req)
    self.assertEqual('foobar', req.headers['X-Auth-Url'])