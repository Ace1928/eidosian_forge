import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_environ_deprecated_variables(self):
    value = uuid.uuid4().hex
    environ = {'HTTP_X_USER': value}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(value, ctx.user)
    environ = {'HTTP_X_TENANT_ID': value}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(value, ctx.project_id)
    environ = {'HTTP_X_STORAGE_TOKEN': value}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(value, ctx.auth_token)
    environ = {'HTTP_X_TENANT': value}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(value, ctx.project_id)
    environ = {'HTTP_X_ROLE': value}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual([value], ctx.roles)
    environ = {'HTTP_X_TENANT_NAME': value}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(value, ctx.project_name)