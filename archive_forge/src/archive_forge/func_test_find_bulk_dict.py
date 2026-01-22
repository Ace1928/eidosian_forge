from osc_lib import exceptions
from openstackclient.api import api
from openstackclient.tests.unit.api import fakes as api_fakes
def test_find_bulk_dict(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json={'qaz': api_fakes.LIST_RESP}, status_code=200)
    ret = self.api.find_bulk('qaz', id='1')
    self.assertEqual([api_fakes.LIST_RESP[0]], ret)