import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_create_router_specific_tenant(self):
    new_router_tenant_id = 'project_id_value'
    mock_router_rep = copy.copy(self.mock_router_rep)
    mock_router_rep['tenant_id'] = new_router_tenant_id
    mock_router_rep['project_id'] = new_router_tenant_id
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers']), json={'router': mock_router_rep}, validate=dict(json={'router': {'name': self.router_name, 'admin_state_up': True, 'project_id': new_router_tenant_id}}))])
    self.cloud.create_router(self.router_name, project_id=new_router_tenant_id)
    self.assert_calls()