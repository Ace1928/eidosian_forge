import datetime
from unittest import mock
import uuid
from keystoneauth1 import fixture
import testtools
import webob
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _request
def test_invalid_service_token(self):
    token_id = uuid.uuid4().hex

    @webob.dec.wsgify
    def _do_cb(req):
        self.assertEqual('Invalid', req.headers['X-Service-Identity-Status'])
        self.assertEqual(token_id, req.headers['X-Service-Token'])
        return webob.Response()
    m = FetchingMiddleware(_do_cb)
    self.call(m, headers={'X-Service-Token': token_id})