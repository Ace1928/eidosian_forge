import datetime
from unittest import mock
import uuid
from keystoneauth1 import fixture
import testtools
import webob
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _request
def test_good_v3_service_token(self):
    t = fixture.V3Token()
    t.set_project_scope()
    role = t.add_role()
    token_id = uuid.uuid4().hex
    token_dict = {token_id: t}

    @webob.dec.wsgify
    def _do_cb(req):
        self.assertEqual(token_id, req.headers['X-Service-Token'].strip())
        self.assertEqual('Confirmed', req.headers['X-Service-Identity-Status'])
        self.assertNotIn('X-Auth-Token', req.headers)
        p = req.environ['keystone.token_auth']
        self.assertFalse(p.has_user_token)
        self.assertTrue(p.has_service_token)
        self.assertEqual(t.project_id, p.service.project_id)
        self.assertEqual(t.project_domain_id, p.service.project_domain_id)
        self.assertEqual(t.user_id, p.service.user_id)
        self.assertEqual(t.user_domain_id, p.service.user_domain_id)
        self.assertIn(role['name'], p.service.role_names)
        return webob.Response()
    m = FetchingMiddleware(_do_cb, token_dict)
    self.call(m, headers={'X-Service-Token': token_id})
    self.call(m, headers={'X-Service-Token': token_id + ' '})
    self.call(m, headers={'X-Service-Token': token_id + '\r'})