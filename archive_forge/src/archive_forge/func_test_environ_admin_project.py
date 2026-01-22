import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_environ_admin_project(self):
    environ = {}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertIs(True, ctx.is_admin_project)
    self.assertIs(True, ctx.to_policy_values()['is_admin_project'])
    environ = {'HTTP_X_IS_ADMIN_PROJECT': 'True'}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertIs(True, ctx.is_admin_project)
    self.assertIs(True, ctx.to_policy_values()['is_admin_project'])
    environ = {'HTTP_X_IS_ADMIN_PROJECT': 'False'}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertIs(False, ctx.is_admin_project)
    self.assertIs(False, ctx.to_policy_values()['is_admin_project'])