from osc_lib import exceptions
from openstackclient.api import api
from openstackclient.tests.unit.api import fakes as api_fakes
def test_create_put(self):
    self.requests_mock.register_uri('PUT', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=202)
    ret = self.api.create('qaz', method='PUT')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret)