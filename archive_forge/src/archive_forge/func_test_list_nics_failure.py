from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_nics_failure(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail']), status_code=400)])
    self.assertRaises(exceptions.SDKException, self.cloud.list_nics)
    self.assert_calls()