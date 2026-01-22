import fixtures
import testresources
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib import simple_cert
def test_list_ca_certificates(self):
    body = {'certificates': [{'name': 'admin'}, {'name': 'admin2'}]}
    get_mock = self._mock_request_method(method='get', body=body)
    response = self.mgr.get_ca_certificates()
    self.assertEqual(response.request_ids[0], self.TEST_REQUEST_ID)
    get_mock.assert_called_once_with('/OS-SIMPLE-CERT/ca', authenticated=False)