from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_hypervisor_search(self):
    expected = [dict(id=self.data_fixture.hyper_id_1, hypervisor_hostname='hyper1', state='up', status='enabled'), dict(id=self.data_fixture.hyper_id_2, hypervisor_hostname='hyper2', state='up', status='enabled')]
    result = self.cs.hypervisors.search('hyper')
    self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
    if self.cs.api_version >= api_versions.APIVersion('2.53'):
        self.assert_called('GET', '/os-hypervisors?hypervisor_hostname_pattern=hyper')
    else:
        self.assert_called('GET', '/os-hypervisors/hyper/search')
    for idx, hyper in enumerate(result):
        self.compare_to_expected(expected[idx], hyper)