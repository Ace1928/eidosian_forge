from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_get_no_response_key(self):
    self.response.json.return_value = {'id': 42}
    expected = HumanResource(self.manager, {'id': 42}, loaded=True)
    result = self.manager._get('/human_resources/42')
    self.manager.client.get.assert_called_with('/human_resources/42')
    self.assertEqual(expected, result)