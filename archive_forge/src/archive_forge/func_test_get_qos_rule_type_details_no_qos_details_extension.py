from openstack import exceptions
from openstack.network.v2 import qos_rule_type
from openstack.tests.unit import base
def test_get_qos_rule_type_details_no_qos_details_extension(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension]})])
    self.assertRaises(exceptions.SDKException, self.cloud.get_qos_rule_type_details, self.rule_type_name)
    self.assert_calls()