from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_hypervisor_index(self):
    expected = [dict(id=self.data_fixture.hyper_id_1, hypervisor_hostname='hyper1'), dict(id=self.data_fixture.hyper_id_2, hypervisor_hostname='hyper2')]
    result = self.cs.hypervisors.list(False)
    self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/os-hypervisors')
    for idx, hyper in enumerate(result):
        self.compare_to_expected(expected[idx], hyper)