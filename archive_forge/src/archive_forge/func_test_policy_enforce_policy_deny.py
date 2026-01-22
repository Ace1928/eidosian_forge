from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
@mock.patch.object(policy.Enforcer, 'enforce')
def test_policy_enforce_policy_deny(self, mock_enforce):
    mock_enforce.return_value = False
    self.assertRaises(exc.HTTPForbidden, self.controller.an_action, self.req, tenant_id='foo')