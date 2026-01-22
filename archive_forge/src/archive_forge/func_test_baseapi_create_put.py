from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_create_put(self):
    self.requests_mock.register_uri('PUT', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=202)
    ret = self.api.create('qaz', method='PUT')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret)