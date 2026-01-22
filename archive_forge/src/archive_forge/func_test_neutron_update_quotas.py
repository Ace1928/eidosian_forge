from openstack import exceptions
from openstack.network.v2 import quota as _quota
from openstack.tests.unit import base
def test_neutron_update_quotas(self):
    project = self.mock_for_keystone_projects(project_count=1, list_get=True)[0]
    self.register_uris([dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'quotas', project.project_id]), json={}, validate=dict(json={'quota': {'network': 1}}))])
    self.cloud.set_network_quotas(project.project_id, network=1)
    self.assert_calls()