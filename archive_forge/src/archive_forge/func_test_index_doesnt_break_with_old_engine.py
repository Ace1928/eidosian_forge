import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
@mock.patch.object(rpc_client.EngineClient, 'count_stacks')
def test_index_doesnt_break_with_old_engine(self, mock_count_stacks, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    params = {'with_count': 'True'}
    req = self._get('/stacks', params=params)
    engine = self.controller.rpc_client
    engine.list_stacks = mock.Mock(return_value=[])
    mock_count_stacks.side_effect = AttributeError('Should not exist')
    result = self.controller.index(req, tenant_id=self.tenant)
    self.assertNotIn('count', result)