from openstack import exceptions
from openstack.network.v2 import quota as _quota
from openstack.tests.unit import base
def test_neutron_get_quotas(self):
    quota = {'subnet': 100, 'network': 100, 'floatingip': 50, 'subnetpool': -1, 'security_group_rule': 100, 'security_group': 10, 'router': 10, 'rbac_policy': 10, 'port': 500}
    project = self.mock_for_keystone_projects(project_count=1, id_get=True)[0]
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'quotas', project.project_id]), json={'quota': quota})])
    received_quota = self.cloud.get_network_quotas(project.project_id).to_dict(computed=False)
    expected_quota = _quota.Quota(**quota).to_dict(computed=False)
    received_quota.pop('id')
    received_quota.pop('name')
    expected_quota.pop('id')
    expected_quota.pop('name')
    self.assertDictEqual(expected_quota, received_quota)
    self.assert_calls()