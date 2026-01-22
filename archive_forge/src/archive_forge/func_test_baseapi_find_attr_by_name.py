from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_find_attr_by_name(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?name=alpha', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
    ret = self.api.find_attr('qaz', 'alpha')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret)
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?name=0', json={'qaz': []}, status_code=200)
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?id=0', json={'qaz': []}, status_code=200)
    self.assertRaises(exceptions.CommandError, self.api.find_attr, 'qaz', '0')
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz?status=UP', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
    ret = self.api.find_attr('qaz', 'UP', attr='status')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret)
    ret = self.api.find_attr('qaz', value='UP', attr='status')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret)