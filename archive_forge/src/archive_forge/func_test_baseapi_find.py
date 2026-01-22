from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_find(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz/1', json={'qaz': api_fakes.RESP_ITEM_1}, status_code=200)
    ret = self.api.find('qaz', '1')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret)
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz/1', status_code=404)
    self.assertRaises(exceptions.NotFound, self.api.find, 'qaz', '1')