import datetime
from unittest import mock
import uuid
from keystoneauth1 import fixture
import testtools
import webob
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _request
def test_invalid_user_token(self):
    token_id = uuid.uuid4().hex

    @webob.dec.wsgify
    def _do_cb(req):
        self.assertEqual('Invalid', req.headers['X-Identity-Status'])
        self.assertEqual(token_id, req.headers['X-Auth-Token'])
        return webob.Response()
    m = FetchingMiddleware(_do_cb)
    self.call(m, headers={'X-Auth-Token': token_id})