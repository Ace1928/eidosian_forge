from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_endpoint_no_endpoint(self):
    x_api = api.BaseAPI(session=self.sess)
    self.assertIsNotNone(x_api.session)
    self.assertEqual(self.sess, x_api.session)
    self.assertIsNone(x_api.endpoint)
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=200)
    self.assertRaises(ksa_exceptions.EndpointNotFound, x_api._request, 'GET', '/qaz')
    self.assertRaises(ksa_exceptions.EndpointNotFound, x_api._request, 'GET', 'qaz')
    self.assertRaises(ksa_exceptions.connection.UnknownConnectionError, x_api._request, 'GET', '//qaz')