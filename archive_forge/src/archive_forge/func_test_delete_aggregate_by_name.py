from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_aggregate_by_name(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', self.aggregate_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-aggregates']), json={'aggregates': [self.fake_aggregate]}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1']))])
    self.assertTrue(self.cloud.delete_aggregate(self.aggregate_name))
    self.assert_calls()