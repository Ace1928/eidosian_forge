import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def test_auth_context_app_cred_with_rule(self):

    def application(environ, start_response):
        body = b'body'
        headers = [('Content-Type', 'text/html; charset=utf8'), ('Content-Length', str(len(body)))]
        start_response('200 OK', headers)
        return [body]
    token = self.get_application_credentials_token(self.app_cred_r_id, self.app_cred_r_secret)
    app = webtest.TestApp(auth_context.AuthContextMiddleware(application))
    resp = app.get('/v3/projects/e3a0883d15ff409e98e59d460f583a68', headers={authorization.AUTH_TOKEN_HEADER: token}, status=401)
    self.assertEqual('401 Unauthorized', resp.status)
    app = webtest.TestApp(auth_context.AuthContextMiddleware(application))
    resp = app.get('/v3/users/3879328537914be2b394ddf57a4fc73a', headers={authorization.AUTH_TOKEN_HEADER: token})
    self.assertEqual('200 OK', resp.status)
    self.assertEqual(b'body', resp.body)