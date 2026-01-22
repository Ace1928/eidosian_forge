from openstack import exceptions
from openstack.network.v2 import quota as _quota
from openstack.tests.unit import base
def test_neutron_get_quotas_details(self):
    quota_details = {'subnet': {'limit': 100, 'used': 7, 'reserved': 0}, 'network': {'limit': 100, 'used': 6, 'reserved': 0}, 'floatingip': {'limit': 50, 'used': 0, 'reserved': 0}, 'subnetpool': {'limit': -1, 'used': 2, 'reserved': 0}, 'security_group_rule': {'limit': 100, 'used': 4, 'reserved': 0}, 'security_group': {'limit': 10, 'used': 1, 'reserved': 0}, 'router': {'limit': 10, 'used': 2, 'reserved': 0}, 'rbac_policy': {'limit': 10, 'used': 2, 'reserved': 0}, 'port': {'limit': 500, 'used': 7, 'reserved': 0}}
    project = self.mock_for_keystone_projects(project_count=1, id_get=True)[0]
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'quotas', project.project_id, 'details']), json={'quota': quota_details})])
    received_quota_details = self.cloud.get_network_quotas(project.project_id, details=True)
    self.assertDictEqual(_quota.QuotaDetails(**quota_details).to_dict(computed=False), received_quota_details.to_dict(computed=False))
    self.assert_calls()