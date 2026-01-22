import datetime
from unittest import mock
import uuid
from keystoneauth1 import fixture
import testtools
import webob
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _request
def test_base_doesnt_block_project_id(self):
    project_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    body = uuid.uuid4().hex

    @webob.dec.wsgify
    def _do_cb(req):
        self.assertEqual(project_id, req.headers['X-Project-Id'])
        self.assertEqual(domain_id, req.headers['X-Domain-Id'])
        return webob.Response(body, 200)
    m = FetchingMiddleware(_do_cb)
    resp = self.call(m, headers={'X-Project-Id': project_id, 'X-Domain-Id': domain_id})
    self.assertEqual(body, resp.text)