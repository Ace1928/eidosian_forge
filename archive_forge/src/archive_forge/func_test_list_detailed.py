from osc_lib import exceptions
from openstackclient.api import api
from openstackclient.tests.unit.api import fakes as api_fakes
def test_list_detailed(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz/details', json=api_fakes.LIST_RESP, status_code=200)
    ret = self.api.list('qaz', detailed=True)
    self.assertEqual(api_fakes.LIST_RESP, ret)