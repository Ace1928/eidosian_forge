from osc_lib import exceptions
from openstackclient.api import api
from openstackclient.tests.unit.api import fakes as api_fakes
def test_find_attr_path_resource(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/wsx?name=1', json={'qaz': []}, status_code=200)
    self.requests_mock.register_uri('GET', self.BASE_URL + '/wsx?id=1', json={'qaz': [api_fakes.RESP_ITEM_1]}, status_code=200)
    ret = self.api.find_attr('wsx', '1', resource='qaz')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret)