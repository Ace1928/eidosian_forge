import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_policy_deprecations(self):
    user = uuid.uuid4().hex
    user_domain = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    project_domain = uuid.uuid4().hex
    roles = [uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex]
    ctx = context.RequestContext(user=user, user_domain=user_domain, project_id=project_id, project_domain=project_domain, roles=roles)
    policy = ctx.to_policy_values()
    key = uuid.uuid4().hex
    val = uuid.uuid4().hex
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        policy[key] = val
        self.assertEqual(0, len(w))
        self.assertIs(val, policy[key])
        self.assertEqual(1, len(w))
        self.assertIn(key, str(w[0].message))