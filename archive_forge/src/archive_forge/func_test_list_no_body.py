from osc_lib import exceptions
from openstackclient.api import api
from openstackclient.tests.unit.api import fakes as api_fakes
def test_list_no_body(self):
    self.requests_mock.register_uri('GET', self.BASE_URL, json=api_fakes.LIST_RESP, status_code=200)
    ret = self.api.list('')
    self.assertEqual(api_fakes.LIST_RESP, ret)
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.LIST_RESP, status_code=200)
    ret = self.api.list('qaz')
    self.assertEqual(api_fakes.LIST_RESP, ret)