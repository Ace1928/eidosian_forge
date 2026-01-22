from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_aggregate(self):
    self.register_uris([dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1']))])
    self.assertTrue(self.cloud.delete_aggregate('1'))
    self.assert_calls()