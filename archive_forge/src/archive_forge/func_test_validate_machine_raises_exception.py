import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_validate_machine_raises_exception(self):
    validate_return = {'deploy': {'result': False, 'reason': 'error!'}, 'power': {'result': True, 'reason': None}, 'foo': {'result': True}}
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'validate']), json=validate_return)])
    self.assertRaises(exceptions.ValidationException, self.cloud.validate_machine, self.fake_baremetal_node['uuid'])
    self.assert_calls()