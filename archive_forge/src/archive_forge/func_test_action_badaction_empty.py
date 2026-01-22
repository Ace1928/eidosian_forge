import json
from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.actions as actions
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_action_badaction_empty(self, mock_enforce):
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    body = {}
    req = self._post(stack_identity._tenant_path() + '/actions', data=json.dumps(body))
    self.assertRaises(webob.exc.HTTPBadRequest, self.controller.action, req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, body=body)