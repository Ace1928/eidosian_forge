import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_dict_overrides(self):
    dct = {'auth_token': 'token1', 'user': 'user1', 'read_only': True, 'roles': 'role1,role2,role3', 'color': 'red', 'unknown': ''}
    ctx = context.RequestContext.from_dict(dct, user='user2', project_name='project1')
    self.assertEqual('token1', ctx.auth_token)
    self.assertEqual('user2', ctx.user)
    self.assertEqual('project1', ctx.project_name)
    self.assertIsNone(ctx.project_id)
    self.assertFalse(ctx.is_admin)
    self.assertTrue(ctx.read_only)