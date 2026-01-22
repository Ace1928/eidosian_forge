from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_aggregate(self):
    create_aggregate = self.fake_aggregate.copy()
    del create_aggregate['metadata']
    del create_aggregate['hosts']
    self.register_uris([dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-aggregates']), json={'aggregate': create_aggregate}, validate=dict(json={'aggregate': {'name': self.aggregate_name, 'availability_zone': None}}))])
    self.cloud.create_aggregate(name=self.aggregate_name)
    self.assert_calls()